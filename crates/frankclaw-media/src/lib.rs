#![forbid(unsafe_code)]

mod fetch;
mod store;
pub mod understanding;
pub mod virustotal;

pub use fetch::SafeFetcher;
pub use store::{MediaStore, StoredMediaContent};
