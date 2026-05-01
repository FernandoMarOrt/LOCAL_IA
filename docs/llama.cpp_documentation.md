# Documentación de llama.cpp

## Introducción

[llama.cpp](https://github.com/ggml-org/llama.cpp) es una implementación de inferencia de LLMs en C/C++ con configuración mínima y rendimiento de última generación en una amplia gama de hardware.

### Características Principales

- Implementación en C/C++ puro
- Optimización para Apple Silicon
- Soporte AVX
- Varios niveles de cuantización
- Kernels personalizados CUDA/HIP/MUSA
- Soporte multi-GPU
- Servidor OpenAI-compatible

---

## Instalación

### Requisitos del Sistema

- **CPU**: Soporte AVX2 recomendado
- **GPU**: CUDA, ROCm, Metal (Apple Silicon), o Vulkan
- **Compilador**: GCC 7+ o Clang 5+
- **Build tools**: CMake 3.18+, Make/Ninja

### Compilación desde Fuente

```bash
# Clonar el repositorio
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Compilar con Make
make

# Compilar con CMake
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Ejecutables Disponibles

| Ejecutable | Descripción |
|------------|-------------|
| `llama-cli` | CLI básico para inferencia |
| `llama-server` | Servidor OpenAI-compatible |
| `llama-bench` | Benchmark de rendimiento |
| `llama-quantize` | Cuantización de modelos |
| `llama-tokenize` | Tokenización de texto |
| `llama-chat` | Chat interactivo |

---

## API Reference

### Estructuras Principales

#### `llama_model`

Representa un modelo de lenguaje cargado en memoria.

```cpp
struct llama_model;
```

#### `llama_context`

Contiene el cache KV y estado de cómputo para inferencia.

```cpp
struct llama_context;
```

#### `llama_vocab`

Vocabulario del modelo para tokenización y detokenización.

```cpp
struct llama_vocab;
```

#### `llama_sampler`

Controla la selección de tokens durante la generación.

```cpp
struct llama_sampler;
```

---

## Funciones Principales

### Carga de Modelo

```cpp
// Parámetros para cargar el modelo
llama_model_params mparams = llama_model_default_params();
mparams.n_gpu_layers = 99;  // Todas las capas en GPU

// Cargar modelo desde archivo
llama_model *model = llama_model_load_from_file("model.gguf", mparams);
```

### Creación de Contexto

```cpp
// Parámetros del contexto
llama_context_params cparams = llama_context_default_params();
cparams.n_ctx = 2048;      // Tamaño del contexto
cparams.n_batch = 512;     // Tamaño máximo de lote
cparams.n_threads = 8;     // Hilos para generación
cparams.n_threads_batch = 8;  // Hilos para batch

// Inicializar contexto desde modelo
llama_context *ctx = llama_init_from_model(model, cparams);
```

### Tokenización

```cpp
#include <string>
#include <vector>

const std::string prompt = "Once upon a time";
int n = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
std::vector<llama_token> tokens(n);
llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), n, true, true);
```

### Cadena de Muestreo (Sampler Chain)

```cpp
// Crear cadena de samplers
auto sparams = llama_sampler_chain_default_params();
llama_sampler *smpl = llama_sampler_chain_init(sparams);

// Añadir samplers individuales
llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
```

### Decodificación

```cpp
// Crear batch con tokens
llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

// Decodificar
llama_decode(ctx, batch);
```

### Generación de Tokens

```cpp
for (int i = 0; i < 200; i++) {
    llama_token id = llama_sampler_sample(smpl, ctx, -1);
    
    if (llama_vocab_is_eog(vocab, id)) {
        break;  // End of generation
    }
    
    char buf[64];
    int len = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
    printf("%.*s", len, buf);
    fflush(stdout);
    
    batch = llama_batch_get_one(&id, 1);
    llama_decode(ctx, batch);
}
```

### Limpieza de Recursos

```cpp
llama_sampler_free(smpl);
llama_free(ctx);
llama_model_free(model);
llama_backend_free();
```

---

## Parámetros de Muestreo

### Parámetros Disponibles

| Parámetro | Tipo | Descripción | Valor por defecto |
|-----------|------|-------------|-------------------|
| `--samplers` | string | Lista ordenada de samplers (separados por `;`) | - |
| `--seed` | integer | Semilla RNG para reproducibilidad | -1 |
| `--temperature` | float | Temperatura de muestreo | 0.80 |
| `--top-k` | integer | Límite top-k sampling | 40 |
| `--top-p` | float | Límite top-p (nucleus) sampling | 0.95 |
| `--min-p` | float | Límite min-p sampling | 0.05 |
| `--repeat-penalty` | float | Penalización por repetición | 1.00 |
| `--mirostat` | integer | Habilitar Mirostat (0=off, 1=M1, 2=M2) | 0 |
| `--grammar` | string | Gramática BNF para restringir salida | - |
| `--json-schema` | string | Esquema JSON para estructura de salida | - |

### Ejemplo de Configuración de Samplers

```cpp
llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
    64,    // penalty_last_n: penalizar últimos 64 tokens
    1.1f,  // penalty_repeat
    0.0f,  // penalty_freq
    0.0f   // penalty_present
));
```

---

## Parámetros CLI

### Configuración General

| Parámetro | Descripción |
|-----------|-------------|
| `-h, --help` | Mostrar ayuda y salir |
| `--version` | Mostrar versión |
| `--license` | Mostrar licencia y dependencias |
| `-cl, --cache-list` | Listar modelos en caché |

### CPU y Threading

| Parámetro | Descripción |
|-----------|-------------|
| `-t, --threads N` | Número de hilos CPU (default: -1) |
| `-tb, --threads-batch N` | Hilos para batch/prompt processing |
| `-C, --cpu-mask M` | Máscara de afinidad CPU (hex) |
| `-Cr, --cpu-range lo-hi` | Rango de CPUs para afinidad |
| `--prio N` | Prioridad del proceso (-1=low, 0=normal, 1=medium, 2=high, 3=realtime) |

### Contexto y Generación

| Parámetro | Descripción |
|-----------|-------------|
| `-c, --ctx-size N` | Tamaño del contexto del prompt |
| `-n, --predict N` | Número de tokens a predecir (default: -1) |
| `-b, --batch-size N` | Tamaño máximo lógico del batch |
| `-ub, --ubatch-size N` | Tamaño máximo físico del batch |
| `--keep N` | Número de tokens a mantener del prompt inicial |

### Escalado RoPE y YaRN

| Parámetro | Descripción |
|-----------|-------------|
| `--rope-scaling` | Método de escalado: none, linear, yarn |
| `--rope-scale N` | Factor de escalado del contexto RoPE |
| `--yarn-ext-factor N` | Factor de mezcla de extrapolación YaRN |

### Memoria y Caché

| Parámetro | Descripción |
|-----------|-------------|
| `-ctk, --cache-type-k TYPE` | Tipo de datos KV cache para K (f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1) |
| `-ctv, --cache-type-v TYPE` | Tipo de datos KV cache para V |
| `--mlock` | Forzar modelo a RAM |
| `--mmap, --no-mmap` | Habilitar/deshabilitar memory-mapping |

---

## Ejemplos de Código

### Ejemplo Completo de Generación de Texto

```cpp
#include "llama.h"
#include <cstdio>
#include <string>
#include <vector>

int main() {
    ggml_backend_load_all();

    // 1. Cargar modelo
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99; // Todas las capas en GPU
    llama_model *model = llama_model_load_from_file("model.gguf", mparams);

    const llama_vocab *vocab = llama_model_get_vocab(model);

    // 2. Crear contexto
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 512;
    llama_context *ctx = llama_init_from_model(model, cparams);

    // 3. Tokenizar
    const std::string prompt = "Once upon a time";
    int n = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), n, true, true);

    // 4. Construir sampler
    llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

    // 5. Decodificar prompt
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    llama_decode(ctx, batch);

    // 6. Bucle de generación
    for (int i = 0; i < 200; i++) {
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;

        char buf[64];
        int len = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        printf("%.*s", len, buf);
        fflush(stdout);

        batch = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx, batch) != 0) break;
    }
    printf("\n");

    // 7. Imprimir estadísticas de rendimiento
    llama_perf_context_print(ctx);

    // 8. Limpieza
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
```

### Ejemplo de Ejecución CLI

```bash
./llama-simple -m ./models/llama-7b-v2/ggml-model-f16.gguf "Hello my name is"
```

---

## Cuantización de Modelos

### Tipos de Cuantización

| Tipo | Bits | Descripción |
|------|------|-------------|
| f32 | 32 | Punto flotante de 32 bits (original) |
| f16 | 16 | Punto flotante de 16 bits (FP16) |
| q4_0 | 4 | Cuantización 4 bits estándar |
| q4_1 | 4 | Cuantización 4 bits mejorada |
| q5_0 | 5 | Cuantización 5 bits estándar |
| q5_1 | 5 | Cuantización 5 bits mejorada |
| q8_0 | 8 | Cuantización 8 bits |
| iq2_xs | 2 | Cuantización 2 bits extra pequeña |
| iq3_xs | 3 | Cuantización 3 bits extra pequeña |
| iq4_nl | 4 | Cuantización 4 bits no lineal |
| iq3_s | 3 | Cuantización 3 bits pequeña |
| iq2_s | 2 | Cuantización 2 bits pequeña |

### Comando de Cuantización

```bash
./llama-quantize model-f32.gguf model-q4_0.gguf q4_0
```

---

## Referencias

- [Repositorio oficial](https://github.com/ggml-org/llama.cpp)
- [Documentación del servidor](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [Ejemplos simples](https://github.com/ggml-org/llama.cpp/blob/master/examples/simple/README.md)
- [Completación CLI](https://github.com/ggml-org/llama.cpp/blob/master/tools/completion/README.md)
