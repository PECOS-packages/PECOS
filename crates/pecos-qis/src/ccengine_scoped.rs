//! Caller-stack orchestration. Scheduling and quantum work use shared paths.
use super::{ByteMessage, EngineStage, InterfaceError, PecosError, QisEngine, Shot, ShotLifecycle};

use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ExecutionMode {
    Staged,
    Scoped,
}

impl QisEngine {
    /// Opt into synchronous native execution through `ControlEngine::run_with`.
    ///
    /// The default is staged execution. Scoped execution cannot detach a native
    /// program that never reaches a cooperative cancellation checkpoint. This
    /// choice is copied to Monte Carlo templates, but no live callback is copied.
    /// Direct `start`/`continue_processing` driving is unavailable in this mode.
    ///
    /// # Errors
    /// Rejects changes during a shot or an interface without scoped support.
    pub fn set_scoped_execution(&mut self, enabled: bool) -> Result<(), PecosError> {
        if self.shot_lifecycle == ShotLifecycle::Running
            || self
                .dynamic_state
                .as_ref()
                .is_some_and(|s| !s.execution_complete)
        {
            return Err(PecosError::Input(
                "Cannot change execution mode during a shot".into(),
            ));
        }
        if enabled
            && !self
                .interface
                .as_ref()
                .is_some_and(|i| i.supports_scoped_execution())
        {
            return Err(PecosError::Input(
                "Interface does not support scoped execution".into(),
            ));
        }
        self.execution_mode = if enabled {
            ExecutionMode::Scoped
        } else {
            ExecutionMode::Staged
        };
        Ok(())
    }

    pub(super) fn run_scoped(
        &mut self,
        process: &mut dyn FnMut(ByteMessage) -> Result<ByteMessage, PecosError>,
    ) -> Result<Shot, PecosError> {
        if let Some(error) = self.terminal_failure_error() {
            return Err(error);
        }
        self.validate_shot_start()?;
        let result = catch_unwind(AssertUnwindSafe(|| self.run_scoped_inner(process)));
        // No program/host callback frames are live here. Even startup failure
        // requires whole-host recovery; Clone and a mode switch retain this latch.
        if let Some(state) = &mut self.dynamic_state {
            state.execution_complete = true;
        }
        match result {
            Ok(Ok(shot)) => Ok(shot),
            Ok(Err(error)) => {
                self.latch_terminal_error(error.to_string());
                self.reset_failure.get_or_insert_with(|| error.to_string());
                Err(error)
            }
            Err(payload) => {
                let message = "scoped execution panicked; whole-host reset required".to_owned();
                self.latch_terminal_error(message.clone());
                self.reset_failure.get_or_insert(message);
                resume_unwind(payload)
            }
        }
    }

    fn run_scoped_inner(
        &mut self,
        process: &mut dyn FnMut(ByteMessage) -> Result<ByteMessage, PecosError>,
    ) -> Result<Shot, PecosError> {
        if !self
            .interface
            .as_ref()
            .is_some_and(|i| i.supports_scoped_execution())
        {
            return Err(PecosError::Input(
                "Interface does not support scoped execution".into(),
            ));
        }
        self.prepare_shot()?;
        self.prepare_dynamic_execution()?;
        if self
            .dynamic_state
            .as_ref()
            .and_then(|s| s.sync_handle.as_ref())
            .is_none()
        {
            return Err(PecosError::Input(
                "Scoped execution requires a response handle".into(),
            ));
        }
        // The callback borrows the controller, not the program interface. The
        // interface stays owned here until the guard returns, including panics.
        let mut interface = self.interface.take().expect("prepared interface");
        let mut host_error = None;
        let mut host_panic = None;
        let execution = catch_unwind(AssertUnwindSafe(|| {
            interface.collect_operations_scoped(&mut |id| {
                if host_error.is_some() || host_panic.is_some() {
                    return Err(InterfaceError::Other("Host request already failed".into()));
                }
                match catch_unwind(AssertUnwindSafe(|| {
                    self.service_scoped_request(id, process)
                })) {
                    Ok(Ok(())) => Ok(()),
                    Ok(Err(error)) => {
                        host_error = Some(error);
                        Err(InterfaceError::Other("Scoped host request failed".into()))
                    }
                    Err(payload) => {
                        host_panic = Some(payload);
                        Err(InterfaceError::Other("Scoped host request panicked".into()))
                    }
                }
            })
        }));
        self.interface = Some(interface);
        self.dynamic_state
            .as_mut()
            .expect("prepared state")
            .execution_complete = true;
        // Restore ownership before teardown too: a third-party teardown may panic.
        let teardown = catch_unwind(AssertUnwindSafe(|| {
            self.interface
                .as_mut()
                .expect("restored interface")
                .disable_dynamic_mode()
        }));
        match &teardown {
            Ok(Ok(())) => {}
            Ok(Err(error)) => self.reset_failure = Some(format!("scoped teardown failed: {error}")),
            Err(_) => self.reset_failure = Some("scoped teardown panicked; reset required".into()),
        }
        if let Some(payload) = host_panic {
            resume_unwind(payload);
        }
        if let Some(error) = host_error {
            return Err(error);
        }
        let execution = match execution {
            Ok(result) => result,
            Err(payload) => resume_unwind(payload),
        };
        let collector = execution.map_err(crate::interface_impl::interface_error_to_pecos)?;
        match teardown {
            Ok(result) => result.map_err(crate::interface_impl::interface_error_to_pecos)?,
            Err(payload) => resume_unwind(payload),
        }
        self.pending_dynamic_ops = collector.operations;
        loop {
            match self.completed_stage()? {
                EngineStage::Complete(shot) => return Ok(shot),
                EngineStage::NeedsProcessing(commands) => {
                    self.process_scoped_commands(commands, process)?;
                }
            }
        }
    }

    fn service_scoped_request(
        &mut self,
        id: u64,
        process: &mut dyn FnMut(ByteMessage) -> Result<ByteMessage, PecosError>,
    ) -> Result<(), PecosError> {
        while let Some(commands) = self.process_result_request(id, "scoped_request")? {
            self.process_scoped_commands(commands, process)?;
        }
        Ok(())
    }

    fn process_scoped_commands(
        &mut self,
        mut commands: ByteMessage,
        process: &mut dyn FnMut(ByteMessage) -> Result<ByteMessage, PecosError>,
    ) -> Result<(), PecosError> {
        loop {
            // Do not call a simulator directly or skip empty/timed messages.
            // HybridEngine's callback runs the complete QuantumSystem lifecycle.
            let response = process(commands)?;
            let feedback = self.accept_feedback(&response)?;
            if let Some(next) = feedback.commands {
                commands = next;
            } else {
                // Match staged caching of ALL fresh nonnative outcomes. A later
                // cache hit must not create a callback and split a quantum batch.
                // Staged completion deliberately does not publish terminal results.
                if self
                    .dynamic_state
                    .as_ref()
                    .is_some_and(|s| !s.execution_complete)
                {
                    self.publish_measurement_outcomes(&feedback.updates)?;
                }
                self.measurement_mapping.clear();
                return Ok(());
            }
        }
    }
}
