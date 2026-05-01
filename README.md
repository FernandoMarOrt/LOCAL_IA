# LLAMA CPP — Ejecutar IA en local

Resumen
-------
Este repositorio contiene scripts y recursos para ejecutar un modelo de lenguaje grande (LLM) en local usando binarios compatibles con el formato GGUF/ggml (por ejemplo, builds basadas en llama.cpp / ggml). El objetivo es facilitar la ejecución del modelo en tu máquina (CPU o GPU) con parámetros configurables.

Estructura del proyecto
-----------------------
- `arranque-qwen.sh`: Script de arranque/ejecución para lanzar el modelo con los parámetros principales.
- `models/`: Carpeta donde se colocan los modelos en formato `.gguf` o equivalente. Ejemplo incluido: `Qwen3.5-9B-UD-Q6_K_XL.gguf`.
- `runtime/`: Archivos binarios, compilaciones o utilidades runtime necesarias (puede variar según tu build local).
- `docs/`: Documentación adicional y notas de conversión/uso.
- `archives/`: (Opcional) backups o modelos antiguos.

Requisitos previos
------------------
- Tener compilado el binario compatible (por ejemplo `main` de `llama.cpp` o su fork) en `runtime/` o en tu PATH.
- Modelo en `models/` en formato `.gguf` o el que soporte tu runtime.
- Dependencias del sistema: `bash`, compilador (si compilas), drivers GPU si usarás aceleración (CUDA/ROCm).

Uso básico (ejemplo)
--------------------
1. Coloca el modelo en `models/`, p. ej. `models/Qwen3.5-9B-UD-Q6_K_XL.gguf`.
2. Ejecuta el script de arranque:

```bash
./arranque-qwen.sh
```

Nota: `arranque-qwen.sh` normalmente contiene la llamada al binario con argumentos. Puedes editar ese script o ejecutar el binario directamente con los parámetros que describo abajo.

Parámetros configurables (comunes)
---------------------------------
Los runtimes tipo `llama.cpp` aceptan una serie de flags. A continuación están los parámetros más importantes y su efecto:

- `--model` / `-m`: **Ruta al archivo del modelo**. Ejemplo: `models/Qwen3.5-9B-UD-Q6_K_XL.gguf`.
- `--threads`: **Número de hilos (CPU)** para inferencia. Aumenta throughput en CPU, prueba valores entre 2 y el número de cores disponibles.
- `--n_ctx`: **Tamaño de contexto** (tokens). Controla cuánto contexto mantiene el modelo (p. ej. 2048, 4096, 8192). Mayor contexto consume más memoria.
- `--n_gpu_layers`: **Capas que se ejecutan en GPU** (si el build soporta GPU). Ajusta para equilibrar memoria GPU y CPU.
- `--batch`: **Tamaño de lote** para procesamiento por paso (si aplica).
- `--temp` / `--temperature`: **Aleatoriedad** en generación. Valores típicos: 0.0–1.2. Más alto => salidas más diversas.
- `--top_k`: **Top-K sampling**. Limita selección a los K tokens más probables.
- `--top_p`: **Top-p (nucleus) sampling**. Selecciona tokens hasta alcanzar probabilidad acumulada p.
- `--repeat_penalty`: **Penalización por repetición** para evitar loops repetitivos.
- `--seed`: **Semilla aleatoria** para reproducibilidad.
- `--gpu`: (o flags propios del runtime) **Seleccionar dispositivo GPU** si aplica.

Ejemplo de invocación con parámetros:

```bash
./arranque-qwen.sh --model models/Qwen3.5-9B-UD-Q6_K_XL.gguf --threads 8 --n_ctx 8192 --temperature 0.7 --top_k 40 --top_p 0.95 --repeat_penalty 1.1
```

Dónde cambiar los parámetros
----------------------------
- Edita `arranque-qwen.sh` para ajustar los flags por defecto.
- Alternativamente, ejecuta el binario directamente desde `runtime/` y pásale los parámetros en la línea de comandos.
- Si se usan wrappers o un servicio local, busca archivos de configuración dentro de `runtime/` o `docs/`.

Recomendaciones de hardware y memoria
------------------------------------
- Modelos grandes (p. ej. 7B, 9B, 13B) consumen mucha RAM y/o VRAM. Asegúrate de tener memoria suficiente o usa versiones cuantizadas (`Q4/Q6`) si tu runtime lo soporta.
- Para CPU-only: usa `--threads` y reduce `--n_ctx` para evitar OOM.
- Para GPU: asigna `--n_gpu_layers` según memoria disponible; ten drivers y runtime (CUDA/ROCm) instalados.

Conversión y compatibilidad de modelos
-------------------------------------
Si tienes un modelo en otro formato (PyTorch, HuggingFace), consulta `docs/llama.cpp_documentation.md` para pasos de conversión a `.gguf` y recomendaciones de cuantización. La conversión suele requerir utilidades externas (scripts de conversión y herramientas de quantization).

Debug y solución de problemas
-----------------------------
- Si obtienes errores de memoria, reduce `--n_ctx`, `--n_gpu_layers` o usa una versión más pequeña/quantizada del modelo.
- Si hay errores al ejecutar `arranque-qwen.sh`, ejecuta el binario directamente para ver la salida detallada.
- Revisa `runtime/` para comprobar que el binario es ejecutable y compatible con tu OS.

Buenas prácticas
----------------
- Mantén los modelos grandes fuera del control de versiones y en `models/` local.
- Documenta los cambios de parámetros en `arranque-qwen.sh` y mantiene un ejemplo reproducible.

Publicar en GitHub de forma segura
----------------------------------
Este proyecto ya está preparado para subirlo a GitHub sin exponer variables sensibles:

- Se usa `.env` / `.env.local` para variables de entorno locales.
- Existe `.env.example` como plantilla pública.
- `.gitignore` excluye secretos, modelos, binarios y archivos temporales.

### Flujo recomendado

1. Crear tu archivo local de variables (no versionado):

```bash
cp .env.example .env
```

2. Editar `.env` y definir una clave segura en `LLAMA_API_KEY`.

3. Verificar que Git está ignorando secretos y binarios:

```bash
git check-ignore -v .env .env.local models runtime archives
```

Si esas carpetas ya estaban versionadas antes de crear `.gitignore`, sácalas del índice sin borrarlas de disco:

```bash
git rm -r --cached models runtime archives
```

4. Confirmar qué archivos se subirán:

```bash
git status
git add .
git status
```

5. Crear commit y subir al repositorio remoto:

```bash
git commit -m "docs: preparar proyecto para publicacion segura en GitHub"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
git push -u origin main
```

### Si alguna clave ya se subió por error

- Rota/revoca inmediatamente la clave comprometida.
- Elimina el secreto del historial de Git antes de seguir compartiendo el repo.
- Recomendado: activar Secret Scanning y Push Protection en GitHub.

Referencias oficiales
---------------------
- `llama.cpp` (repositorio oficial): https://github.com/ggml-org/llama.cpp
- Documentación de servidor (`llama-server`): https://github.com/ggml-org/llama.cpp/blob/master/docs/server.md
- Formato GGUF: https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/README.md
- GitHub docs sobre variables/secretos en Actions: https://docs.github.com/actions/security-guides/encrypted-secrets
- GitHub Secret Scanning: https://docs.github.com/code-security/secret-scanning/about-secret-scanning

Licencia y atribuciones
-----------------------
Comprueba las licencias del modelo y del runtime antes de distribuir o usar comercialmente. Este repositorio es un contenedor/ayudante para ejecutar modelos en local.

Contacto
-------
Si necesitas que adapte el `README` con información más específica sobre tu hardware o quieres que modifique `arranque-qwen.sh` con parámetros por defecto, dímelo y lo hago.

