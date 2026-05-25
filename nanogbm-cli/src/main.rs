use std::error::Error;
use std::path::PathBuf;

use nanogbm::{Config, DatasetBuilder, GbdtTrainer, Model};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: nanogbm train <csv> <label_col> <model_out> | predict <model> <csv> <out>");
        std::process::exit(1);
    }
    match args[1].as_str() {
        "train" => {
            let csv: PathBuf = args[2].clone().into();
            let label_col: usize = args[3].parse()?;
            let model_out: PathBuf = args[4].clone().into();
            train(csv, label_col, model_out)?;
        }
        "predict" => {
            let model: PathBuf = args[2].clone().into();
            let csv: PathBuf = args[3].clone().into();
            let out: PathBuf = args[4].clone().into();
            predict(model, csv, out)?;
        }
        other => {
            eprintln!("unknown command: {other}");
            std::process::exit(1);
        }
    }
    Ok(())
}

fn train(csv: PathBuf, label_col: usize, model_out: PathBuf) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::ReaderBuilder::new().has_headers(true).from_path(&csv)?;
    let headers = rdr.headers()?.clone();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<f32> = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        let mut row = Vec::with_capacity(headers.len() - 1);
        let mut label: Option<f32> = None;
        for (i, v) in rec.iter().enumerate() {
            if i == label_col {
                label = Some(v.parse()?);
            } else {
                row.push(v.parse().unwrap_or(f64::NAN));
            }
        }
        labels.push(label.expect("missing label"));
        rows.push(row);
    }
    let n_rows = rows.len();
    let n_features = rows[0].len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();

    let mut config = Config::default();
    config.verbose = true;
    let ds = DatasetBuilder::from_rows(&flat, n_rows, n_features, &labels, &config)?;
    let model = GbdtTrainer::new(&config).fit(&ds, None)?;
    model.save(&model_out)?;
    println!("Saved model with {} trees", model.trees.len());
    Ok(())
}

fn predict(model: PathBuf, csv: PathBuf, out: PathBuf) -> Result<(), Box<dyn Error>> {
    let model = Model::load(&model)?;
    let mut rdr = csv::ReaderBuilder::new().has_headers(true).from_path(&csv)?;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        let row: Vec<f64> = rec.iter().map(|v| v.parse().unwrap_or(f64::NAN)).collect();
        rows.push(row);
    }
    let n_rows = rows.len();
    let n_features = rows[0].len();
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    let preds = nanogbm::predict::predict_proba(&model, &flat, n_rows, n_features);
    let mut w = csv::Writer::from_path(&out)?;
    w.write_record(["pred"])?;
    for p in preds {
        w.write_record([format!("{p}")])?;
    }
    w.flush()?;
    Ok(())
}
