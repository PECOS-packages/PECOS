# pecos-bp

Native belief-propagation primitives for PECOS decoders. This crate owns reusable BP graph construction, scratch storage, and the native min-sum implementation.

It is intended to grow to include native Relay-BP, layered schedules, and a GPU BP kernel while keeping those primitives available across decoder implementations.
