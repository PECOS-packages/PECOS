//! Lexical execution context for a logical Monte Carlo worker.
use pecos_core::errors::PecosError;

/// Supplies an execution context without adding framework dependencies to engines.
///
/// Call `work` synchronously, exactly once, on the calling thread. The callback
/// owns the worker engine, including its destruction. A scope may reject entry
/// with an error; in that case the unexecuted engine is dropped without entering
/// the context. Never retain live guards in cloneable engine configuration.
/// Each nonempty logical worker job gets its own invocation, not one invocation
/// per physical pool thread. Scopes must clean up on errors and unwinding.
///
/// The host rejects skipped work and execution on a different thread, preserves
/// work errors, and translates unwinding panics to processing errors.
pub trait WorkerScope: Send + Sync {
    /// Enter the context, execute work, and leave the context on this thread.
    ///
    /// # Errors
    /// Return an error if context entry or teardown fails.
    fn execute(&self, work: Box<dyn FnOnce() + Send + '_>) -> Result<(), PecosError>;
}

/// Existing execution behavior: invoke the work without an additional context.
pub struct DirectWorkerScope;

impl WorkerScope for DirectWorkerScope {
    fn execute(&self, work: Box<dyn FnOnce() + Send + '_>) -> Result<(), PecosError> {
        work();
        Ok(())
    }
}

pub(super) fn execute_scoped(
    scope: &dyn WorkerScope,
    work: impl FnOnce() -> Result<(), PecosError> + Send,
) -> Result<(), PecosError> {
    let thread = std::thread::current().id();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut outcome = None;
        let scope_result = scope.execute(Box::new(|| {
            outcome = Some(if std::thread::current().id() == thread {
                work()
            } else {
                Err(PecosError::Processing(
                    "Worker scope moved work to another thread".into(),
                ))
            });
        }));
        if let Some(result) = outcome {
            result?;
            scope_result
        } else {
            scope_result?;
            Err(PecosError::Processing(
                "Worker scope did not execute work".into(),
            ))
        }
    }));
    result.unwrap_or_else(|payload| {
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("unknown panic");
        Err(PecosError::Processing(format!(
            "Worker scope failed: {message}"
        )))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Swallow;
    impl WorkerScope for Swallow {
        fn execute(&self, work: Box<dyn FnOnce() + Send + '_>) -> Result<(), PecosError> {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work));
            Ok(())
        }
    }
    #[test]
    fn scopes_cannot_hide_work_errors_or_swallowed_panics() {
        let error = execute_scoped(&Swallow, || {
            Err(PecosError::Processing("work failed".into()))
        })
        .unwrap_err();
        assert!(error.to_string().contains("work failed"));
        assert!(execute_scoped(&Swallow, || panic!("work panic")).is_err());
    }
}
