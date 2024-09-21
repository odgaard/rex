use libc::{c_char, c_int};
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use chrono::NaiveDateTime;
use pyo3::prelude::*;
use zstd::stream::read::Decoder;

use arrow::array::{StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_writer::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

const CHUNK_SIZE: usize = 67108864;
const TOTAL_SAMPLE_LINES: usize = 3000000;
const ROW_GROUP_SIZE: usize = 100000;
const ROW_GROUPS_PER_FILE: usize = 10;
const OUTLIER_THRESHOLD: usize = 1000;

fn get_type(query: &str) -> i32 {
    const SYMBOL_TY: i32 = 32;
    // Define the CharTable as per the C++ code
    const CHAR_TABLE: [i32; 128] = [
        // 0-31: Control characters and spaces
        32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, // 32-47: Symbols
        32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
        // 48-57: '0'-'9' (Digits)
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 58-63: Symbols
        32, 32, 32, 32, 32, 32, // 64: '@' (Special character)
        4,  // 65-90: 'A'-'Z' (Uppercase letters)
        2, 2, 2, 2, 2, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        // 91-96: Symbols
        32, 32, 32, 32, 32, 32, // 97-122: 'a'-'z' (Lowercase letters)
        4, 4, 4, 4, 4, 4, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, // 123-127: Symbols
        32, 32, 32, 32, 32,
    ];

    let mut type_accum: i32 = 0;

    for c in query.chars() {
        let code = c as u32;
        let char_type = if code < 128 {
            CHAR_TABLE[code as usize]
        } else {
            SYMBOL_TY
        };
        type_accum |= char_type;
    }

    type_accum
}

fn get_all_types(type_value: i32) -> Vec<i32> {
    let mut types = Vec::new();
    for i in 1..=63 {
        if (type_value & i) == type_value {
            types.push(i);
        }
    }
    types
}

extern "C" {
    fn trainer_wrapper(sample_str: *const c_char, output_path: *const c_char) -> c_int;
    fn compressor_wrapper(
        chunk: *const c_char,
        output_path: *const c_char,
        template_path: *const c_char,
        prefix: c_int,
    ) -> c_int;
}

fn trainer_wrapper_rust(sample_str: &str, output_path: &str) -> PyResult<()> {
    let sample_str_c = CString::new(sample_str)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let output_path_c = CString::new(output_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    unsafe {
        let result = trainer_wrapper(sample_str_c.as_ptr(), output_path_c.as_ptr());
        if result != 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "trainer_wrapper_c failed",
            ));
        }
    }
    Ok(())
}

fn compressor_wrapper_rust(
    chunk: &str,
    output_path: &str,
    template_path: &str,
    prefix: i32,
) -> PyResult<()> {
    let chunk_c = CString::new(chunk)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let output_path_c = CString::new(output_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    let template_path_c = CString::new(template_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    unsafe {
        let result = compressor_wrapper(
            chunk_c.as_ptr(),
            output_path_c.as_ptr(),
            template_path_c.as_ptr(),
            prefix as c_int,
        );
        if result != 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "compressor_wrapper_c failed",
            ));
        }
    }
    Ok(())
}

fn get_variable_info(
    total_chunks: usize,
    group_number: usize,
) -> PyResult<(
    HashMap<usize, HashSet<(i32, i32)>>,
    HashMap<i32, Vec<(i32, i32)>>,
)> {
    let mut variable_to_type = HashMap::new();
    let mut chunk_variables: HashMap<usize, HashSet<(i32, i32)>> = HashMap::new();
    let mut eid_to_variables: HashMap<i32, Vec<(i32, i32)>> = HashMap::new();

    for chunk in 0..total_chunks {
        let variable_tag_file = format!("compressed/{}/variable_{}_tag.txt", group_number, chunk);
        let file = File::open(variable_tag_file)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            let mut parts = line.split_whitespace();
            let variable_str = parts.next().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid variable string")
            })?;
            let tag = parts
                .next()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid tag"))?
                .parse::<i32>()?;

            let mut var_parts = variable_str.split('_');
            let a = var_parts
                .next()
                .and_then(|s| s.strip_prefix("V"))
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid variable format")
                })?
                .parse::<i32>()?;
            let b = var_parts
                .next()
                .and_then(|s| s.strip_prefix("V"))
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid variable format")
                })?
                .parse::<i32>()?;

            let variable = (a, b);
            variable_to_type.insert(variable, tag);
            chunk_variables.entry(chunk).or_default().insert(variable);
            eid_to_variables.entry(a).or_default().push(variable);
        }
    }

    Ok((chunk_variables, eid_to_variables))
}

fn compress_chunk(
    chunk_file_counter: usize,
    current_chunk: &str,
    template_name: &str,
    group_number: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir_name = format!("variable_{}", chunk_file_counter);
    let tag_name = format!("variable_tag_{}.txt", chunk_file_counter);

    // Remove existing directory and file
    let dir_path = Path::new(&dir_name);
    if dir_path.exists() {
        std::fs::remove_dir_all(&dir_path)?;
    }
    let tag_path = Path::new(&tag_name);
    if tag_path.exists() {
        std::fs::remove_file(&tag_path)?;
    }

    // Create the directory
    std::fs::create_dir_all(&dir_path)?;

    let chunk_filename = format!("compressed/{}/chunk{:04}", group_number, chunk_file_counter);
    compressor_wrapper_rust(
        current_chunk,
        &chunk_filename,
        template_name,
        chunk_file_counter as i32,
    )?;

    // Rename files
    let source_dir = dir_path;
    let target_dir = Path::new("compressed")
        .join(group_number.to_string())
        .join(format!("variable_{}", chunk_file_counter));
    std::fs::rename(&source_dir, &target_dir)?;

    let source_tag = tag_path;
    let target_tag = Path::new("compressed")
        .join(group_number.to_string())
        .join(format!("variable_{}_tag.txt", chunk_file_counter));
    std::fs::rename(&source_tag, &target_tag)?;

    Ok(())
}

fn is_valid_timestamp(timestamp: u64) -> bool {
    let min_valid_timestamp: u64 = 946684800; // January 1, 2000, 00:00:00 UTC
    let max_valid_timestamp: u64 = 2524608000; // January 1, 2050, 00:00:00 UTC
    timestamp >= min_valid_timestamp && timestamp < max_valid_timestamp
}

fn to_pyerr(err: Box<dyn std::error::Error>) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string())
}

fn write_parquet_file(
    parquet_files_prefix: &str,
    table: &RecordBatch,
) -> Result<(), Box<dyn std::error::Error>> {
    let writer_properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();

    let mut parquet_file_counter = 0;
    while std::path::Path::new(&format!(
        "{}{}.parquet",
        parquet_files_prefix, parquet_file_counter
    ))
    .exists()
    {
        parquet_file_counter += 1;
    }

    let parquet_filename = format!("{}{}.parquet", parquet_files_prefix, parquet_file_counter);
    let file = std::fs::File::create(parquet_filename)?;
    let mut writer = ArrowWriter::try_new(file, table.schema(), Some(writer_properties))?;
    writer.write(table)?;
    writer.close()?;

    Ok(())
}

#[pyfunction]
fn compress_logs(
    files: Vec<String>,
    index_name: String,
    group_number: usize,
    timestamp_bytes: usize,
    timestamp_format: String,
    parquet_files_prefix: String,
) -> PyResult<()> {
    let template_prefix = format!("compressed/{}_{}", index_name, group_number);
    let mut samples = Vec::new();
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut global_line_count = 0;
    let mut rng = rand::thread_rng();

    std::fs::create_dir_all(format!("compressed/{}", group_number))?;

    let schema = Schema::new(vec![
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("log", DataType::Utf8, false),
    ]);

    let mut epoch_ts_vector = Vec::new();
    let mut log_vector = Vec::new();

    for file_path in files {
        let file = File::open(&file_path)?;
        let reader: Box<dyn BufRead> =
            if Path::new(&file_path).extension().and_then(|s| s.to_str()) == Some("zst") {
                Box::new(BufReader::new(Decoder::new(file).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string())
                })?))
            } else {
                Box::new(BufReader::new(file))
            };

        let mut last_timestamp = 0;
        let mut found_valid_timestamp = false;

        for line in reader.lines() {
            let line =
                line.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
            if line.is_empty() {
                continue;
            }

            // Attempt to parse the timestamp
            let mut epoch_ts = if line.len() >= timestamp_bytes {
                let extract_timestamp_from_this_line = &line[..timestamp_bytes];
                match NaiveDateTime::parse_from_str(
                    extract_timestamp_from_this_line,
                    &timestamp_format,
                ) {
                    Ok(dt) => dt.timestamp() as u64,
                    Err(_) => last_timestamp,
                }
            } else {
                last_timestamp
            };

            // Check if the timestamp is valid
            if !is_valid_timestamp(epoch_ts) {
                if last_timestamp == 0 {
                    eprintln!("Unable to backfill timestamp for a log line, most likely because the start of a file does not contain valid timestamp");
                    eprintln!("This will lead to wrong extracted timestamps");
                    eprintln!(
                        "Attempted to parse '{}' with '{}'",
                        &line[..std::cmp::min(timestamp_bytes, line.len())],
                        timestamp_format
                    );
                }
                // Use last_timestamp even if it's 0
                epoch_ts = last_timestamp;
            } else {
                // Update last_timestamp with the valid timestamp
                last_timestamp = epoch_ts;
            }
            if samples.len() < TOTAL_SAMPLE_LINES {
                samples.push(line.clone());
            } else {
                let j = rng.gen_range(0..global_line_count);
                if j < TOTAL_SAMPLE_LINES {
                    samples[j] = line.clone();
                }
            }

            current_chunk.push_str(&line);
            current_chunk.push('\n');

            epoch_ts_vector.push(epoch_ts);
            log_vector.push(line);

            // Check if the current chunk has reached the maximum size
            if current_chunk.len() >= CHUNK_SIZE {
                chunks.push(std::mem::take(&mut current_chunk));

                let timestamp_array = UInt64Array::from(epoch_ts_vector.clone());
                let log_array = StringArray::from(log_vector.clone());

                let batch = RecordBatch::try_new(
                    Arc::new(schema.clone()),
                    vec![Arc::new(timestamp_array), Arc::new(log_array)],
                )
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

                write_parquet_file(&parquet_files_prefix, &batch).map_err(to_pyerr)?;

                epoch_ts_vector.clear();
                log_vector.clear();
            }

            global_line_count += 1;
        }
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    let samples_str = samples.join("\n");
    trainer_wrapper_rust(&samples_str, &template_prefix)?;

    for (chunk_index, chunk) in chunks.iter().enumerate() {
        compress_chunk(chunk_index, chunk, &template_prefix, group_number).map_err(to_pyerr)?;
    }

    process_compressed_chunks(&chunks, group_number).map_err(to_pyerr)?;
    Ok(())
}

fn process_compressed_chunks(
    chunks: &[String],
    group_number: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let total_chunks = chunks.len();
    let (chunk_variables, eid_to_variables) = get_variable_info(total_chunks, group_number)?;
    let mut touched_types = std::collections::HashSet::new();

    let mut expanded_items: std::collections::HashMap<i32, Vec<String>> =
        std::collections::HashMap::new();
    let mut expanded_lineno: std::collections::HashMap<i32, Vec<usize>> =
        std::collections::HashMap::new();

    let mut current_line_number = if group_number == 0 {
        0
    } else {
        let previous_group_file = format!("compressed/{}/current_line_number", group_number - 1);
        if Path::new(&previous_group_file).exists() {
            std::fs::read_to_string(previous_group_file)?
                .trim()
                .parse::<usize>()?
        } else {
            eprintln!(
                "Warning: previous group's current_line_number file not found. Starting from 0."
            );
            0
        }
    };

    for chunk in 0..total_chunks {
        let mut variable_files = std::collections::HashMap::new();
        for &variable in chunk_variables
            .get(&chunk)
            .unwrap_or(&std::collections::HashSet::new())
        {
            let file_path = format!(
                "compressed/{}/variable_{}/E{}_V{}",
                group_number, chunk, variable.0, variable.1
            );
            variable_files.insert(
                variable,
                std::io::BufReader::new(std::fs::File::open(file_path)?),
            );
        }

        let chunk_filename = format!("compressed/{}/chunk{:04}.eid", group_number, chunk);
        let eid_file = std::fs::File::open(chunk_filename)?;
        let eid_reader = std::io::BufReader::new(eid_file);

        for line in eid_reader.lines() {
            let eid = line?.parse::<i32>()?;
            if eid < 0 || !eid_to_variables.contains_key(&eid) {
                current_line_number += 1;
                continue;
            }

            let this_variables = eid_to_variables.get(&eid).unwrap();
            let mut type_vars = std::collections::HashMap::new();

            for &variable in this_variables {
                let item = variable_files
                    .get_mut(&variable)
                    .unwrap()
                    .lines()
                    .next()
                    .unwrap()?;
                let t = get_type(&item);
                if t == 0 {
                    eprintln!("WARNING, null variable detected in LogCrisp. {} {} {} This variable is not indexed.", chunk, variable.0, variable.1);
                    continue;
                }
                touched_types.insert(t);
                type_vars.entry(t).or_insert_with(Vec::new).push(item);
            }

            for (&t, items) in &type_vars {
                expanded_items
                    .entry(t)
                    .or_default()
                    .extend(items.iter().cloned());
                expanded_lineno.entry(t).or_default().extend(
                    std::iter::repeat(current_line_number / ROW_GROUP_SIZE).take(items.len()),
                );
            }
            current_line_number += 1;
        }
    }

    // Write current_line_number to a file
    std::fs::write(
        format!("compressed/{}/current_line_number", group_number),
        current_line_number.to_string(),
    )?;

    // Process and write compacted types and outliers
    let mut compacted_type_files = std::collections::HashMap::new();
    let mut compacted_lineno_files = std::collections::HashMap::new();
    let mut outlier_file = std::fs::File::create(format!("compressed/{}/outlier", group_number))?;
    let mut outlier_lineno_file =
        std::fs::File::create(format!("compressed/{}/outlier_lineno", group_number))?;
    let mut outlier_items = Vec::new();
    let mut outlier_lineno = Vec::new();

    for &t in &touched_types {
        if expanded_items[&t].is_empty() {
            return Err(format!(
                "Error in variable extraction. No items detected for type {}",
                t
            )
            .into());
        }

        let mut paired: Vec<_> = expanded_items[&t]
            .iter()
            .zip(expanded_lineno[&t].iter())
            .collect();
        paired.sort_unstable_by(|a, b| a.0.cmp(b.0).then_with(|| a.1.cmp(b.1)));

        let mut compacted_items = Vec::new();
        let mut compacted_lineno = Vec::new();
        let mut last_item = String::new();

        for (item, &lineno) in paired {
            if item != &last_item {
                compacted_items.push(item.clone());
                compacted_lineno.push(vec![lineno]);
                last_item = item.clone();
            } else if lineno != *compacted_lineno.last().unwrap().last().unwrap() {
                compacted_lineno.last_mut().unwrap().push(lineno);
            }
        }

        if compacted_items.len() > OUTLIER_THRESHOLD {
            let type_file = compacted_type_files.entry(t).or_insert_with(|| {
                std::fs::File::create(format!("compressed/{}/compacted_type_{}", group_number, t))
                    .unwrap()
            });
            let lineno_file = compacted_lineno_files.entry(t).or_insert_with(|| {
                std::fs::File::create(format!(
                    "compressed/{}/compacted_type_{}_lineno",
                    group_number, t
                ))
                .unwrap()
            });

            for (item, linenos) in compacted_items.iter().zip(compacted_lineno.iter()) {
                writeln!(type_file, "{}", item)?;
                writeln!(
                    lineno_file,
                    "{}",
                    linenos
                        .iter()
                        .map(|&n| n.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                )?;
            }
        } else {
            outlier_items.extend(compacted_items);
            outlier_lineno.extend(compacted_lineno);
        }
    }

    // Sort and write outliers
    let mut paired: Vec<_> = outlier_items
        .into_iter()
        .zip(outlier_lineno.into_iter())
        .collect();
    paired.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    for (item, linenos) in paired {
        writeln!(outlier_file, "{}", item)?;
        writeln!(
            outlier_lineno_file,
            "{}",
            linenos
                .iter()
                .map(|&n| n.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        )?;
    }

    Ok(())
}

#[pymodule]
#[pyo3(name = "rex")]
fn rex(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress_logs, m)?)?;
    Ok(())
}
