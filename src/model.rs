use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use bincode::config::standard;
use serde::{Deserialize, Serialize};

use crate::dataset::BinMapper;
use crate::error::{Error, Result};
use crate::tree::Tree;

/// A trained GBDT model: ensemble of trees plus boosting metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub init_score: f64,
    pub learning_rate: f64,
    pub n_features: usize,
    pub trees: Vec<Tree>,
    /// Per-feature bin mapper, needed for routing raw-value categorical splits.
    /// `#[serde(default)]` keeps backward compatibility with pre-categorical models.
    #[serde(default)]
    pub bin_mappers: Vec<BinMapper>,
}

impl Model {
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let f = File::create(path)?;
        let mut w = BufWriter::new(f);
        let bytes = bincode::serde::encode_to_vec(self, standard())
            .map_err(|e| Error::Serde(e.to_string()))?;
        w.write_all(&bytes)?;
        w.flush()?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let f = File::open(path)?;
        let mut r = BufReader::new(f);
        let mut buf = Vec::new();
        r.read_to_end(&mut buf)?;
        let (model, _) = bincode::serde::decode_from_slice::<Model, _>(&buf, standard())
            .map_err(|e| Error::Serde(e.to_string()))?;
        Ok(model)
    }

    /// Number of times each feature was used as a split, indexed by feature.
    pub fn feature_importance_split(&self) -> Vec<u32> {
        let mut counts = vec![0u32; self.n_features];
        for tree in &self.trees {
            for node in &tree.nodes {
                counts[node.feature as usize] += 1;
            }
        }
        counts
    }

    /// Total split gain attributed to each feature, indexed by feature.
    pub fn feature_importance_gain(&self) -> Vec<f64> {
        let mut gains = vec![0.0f64; self.n_features];
        for tree in &self.trees {
            for node in &tree.nodes {
                gains[node.feature as usize] += node.gain;
            }
        }
        gains
    }
}
