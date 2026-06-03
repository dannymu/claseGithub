# ============================================================
# 10_objetivo1_v5_2_train_supervised_models.R
# Ejecutar entrenamiento supervisado v5.2
# ============================================================

base_dir <- "/mnt/danny_nas/Doctorado-españa/Tesis-doctorado/analisis-VPN"

python_bin <- path.expand("~/env_cluster/bin/python")

py_script <- file.path(
  base_dir,
  "script",
  "10_objetivo1_v5_2_train_supervised_models.py"
)

if (!file.exists(python_bin)) {
  stop("No existe Python: ", python_bin)
}

if (!file.exists(py_script)) {
  stop("No existe script Python: ", py_script)
}

res <- system2(
  python_bin,
  args = py_script,
  stdout = TRUE,
  stderr = TRUE
)

cat(paste(res, collapse = "\n"), "\n")

status <- attr(res, "status")
if (!is.null(status) && status != 0) {
  stop("Falló el entrenamiento supervisado v5.2")
}