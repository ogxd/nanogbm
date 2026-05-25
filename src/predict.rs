use crate::dataset::BinMapper;
use crate::model::Model;
use crate::objective::Objective;
use crate::objective::binary::sigmoid;
use crate::tree::{SplitKind, Tree, bin_goes_left};

/// Walk a single tree on a raw-value row, using `bin_mappers` to route through
/// the same code path as the bin-encoded predict (so categorical splits work).
#[inline]
fn predict_tree_with_mappers(tree: &Tree, row: &[f64], bin_mappers: &[BinMapper]) -> f64 {
    if tree.nodes.is_empty() {
        return tree.leaf_values[0];
    }
    let mut node_idx: i32 = 0;
    loop {
        let node = &tree.nodes[node_idx as usize];
        let feat = node.feature as usize;
        let v = row[feat];
        let go_left = match &node.kind {
            SplitKind::Numerical {
                threshold_value,
                missing_dir,
                ..
            } => {
                if !v.is_finite() {
                    matches!(missing_dir, crate::tree::MissingDir::Left)
                } else {
                    v <= *threshold_value
                }
            }
            SplitKind::Categorical { .. } => {
                let bin = bin_mappers[feat].value_to_bin(v);
                bin_goes_left(&node.kind, bin)
            }
        };
        let next = if go_left {
            node.left_child
        } else {
            node.right_child
        };
        if next < 0 {
            return tree.leaf_values[(!next) as usize];
        }
        node_idx = next;
    }
}

/// Predict raw additive scores for a row-major feature matrix.
pub fn predict_raw_scores(
    model: &Model,
    features: &[f64],
    n_rows: usize,
    n_features: usize,
) -> Vec<f64> {
    let init = model.init_score;
    let has_mappers = !model.bin_mappers.is_empty();
    (0..n_rows)
        .map(|row| {
            let r = &features[row * n_features..(row + 1) * n_features];
            let mut s = init;
            for tree in &model.trees {
                let v = if has_mappers {
                    predict_tree_with_mappers(tree, r, &model.bin_mappers)
                } else {
                    tree.predict_raw(r)
                };
                s += model.learning_rate * v;
            }
            s
        })
        .collect()
}

/// Predict probabilities (sigmoid of raw scores).
pub fn predict_proba(
    model: &Model,
    features: &[f64],
    n_rows: usize,
    n_features: usize,
) -> Vec<f64> {
    let raw = predict_raw_scores(model, features, n_rows, n_features);
    raw.into_iter().map(sigmoid).collect()
}

/// Apply an objective's output-transform to raw scores in place (convenience).
pub fn convert_with(obj: &dyn Objective, raw: &mut [f64]) {
    let raw_copy = raw.to_vec();
    obj.convert_output(&raw_copy, raw);
}
