use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("shape mismatch: {0}")]
    Shape(String),
    #[error("invalid config: {0}")]
    Config(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serde(String),
}

pub type Result<T> = std::result::Result<T, Error>;
