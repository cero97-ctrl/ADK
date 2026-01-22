# Manual de Operaciones: Estandarización y Despliegue de Agentes de IA con Google ADK

Este manual establece los lineamientos técnicos y operativos fundamentales para el ciclo de vida de 
agentes de inteligencia artificial utilizando el **Agent Development Kit (ADK)** de Google Cloud. 
Como arquitectos de soluciones, nuestra prioridad es garantizar la paridad absoluta entre los entornos 
de desarrollo (Local/Colab) y producción en Vertex AI, asegurando la idempotencia de los despliegues y 
la robustez de los sistemas autónomos mediante la estandarización rigurosa de stacks tecnológicos y protocolos de seguridad.

## 1. Fundamentos y Preparación del Entorno de Desarrollo

La estandarización del entorno no es solo una buena práctica, sino una necesidad estratégica para mitigar riesgos de regresión y asegurar que las capacidades de razonamiento del modelo se mantengan constantes. Un entorno fragmentado introduce variables no controladas en la ejecución de herramientas (tools) y en la gestión de la memoria, lo que puede comprometer la fiabilidad del agente en producción.

### 1.1. Evaluación de Especificaciones de Hardware

El rendimiento de la inferencia y la capacidad de orquestación dependen directamente del hardware subyacente. Mientras que el desarrollo local ofrece mayor privacidad y control, los entornos en la nube como Colab permiten acceso inmediato a aceleración de hardware de alto nivel.

| Componente | Google Colab (Cloud) | Desarrollo Local (Mínimo) | Desarrollo Local (Recomendado) |
| --- | --- | --- | --- |
| CPU | Gestionado (Intel Xeon) | Intel i5 (8ª gen) / AMD eq. | Intel i7/i9 (11ª+ gen) / Ryzen 7/9 |
| GPU | NVIDIA T4 / P100 / V100 | Integrada | NVIDIA RTX 3060+ (8+ GB VRAM) |
| RAM | 12 GB - 25+ GB | 16 GB DDR4 | 32 GB DDR4 o superior |
| Almacenamiento | ~80 GB (SSD) | 256 GB SSD (50 GB libres) | 512 GB NVMe SSD o superior |
| Software GPU | Preinstalado | N/A | CUDA Toolkit 11.8/12.0 + cuDNN |

La elección del hardware impacta la latencia de respuesta; para modelos como Gemini 1.5 Pro, una GPU con alta VRAM permite gestionar ventanas de contexto extensas sin degradación del throughput.

### 1.2. Configuración del Stack de Software

Para entornos Windows, es mandatorio el uso de **WSL2** (Ubuntu 22.04 LTS recomendado) para asegurar la compatibilidad con las librerías nativas de Linux utilizadas en contenedores de producción. El stack debe limitarse a Python 3.9, 3.10 o 3.11, dado que la versión 3.12 presenta incompatibilidades conocidas con ciertas dependencias de Vertex AI. Es indispensable contar con Git 2.30+ para el control de versiones y Docker Desktop para los flujos de trabajo de contenerización.

### 1.3. Instalación de Dependencias Críticas

El ecosistema ADK requiere componentes base y librerías de procesamiento de datos para la experimentación avanzada. Ejecute el siguiente bloque en su **entorno virtual**:

```
# Actualización de core y plataforma
pip install --upgrade pip
pip install google-cloud-aiagent google-cloud-aiplatform>=1.38

# Ecosistema de LLMs y Vertex AI
pip install langchain google-generativeai langchain-google-vertexai

# Procesamiento de datos y experimentación
pip install pandas numpy matplotlib jupyter notebook
```
Una vez finalizada la instalación, valide la integridad del entorno consultando la versión del ADK: `python -c "import google.cloud.aiagent as aiagent; print(f'ADK Version: {aiagent.__version__}')"`.

## 2. Protocolo de Seguridad y Gestión de Credenciales en GCP

La gestión de identidades y accesos (IAM) es el perímetro de seguridad más crítico en operaciones de IA. La filtración de una clave de cuenta de servicio no solo compromete la privacidad de los datos, sino que permite el uso no autorizado de recursos de cómputo de alto costo. Adoptamos el principio de **mínimo privilegio** para restringir el alcance de cada agente a sus funciones estrictamente necesarias.

### 2.1. Configuración del Proyecto y APIs

El aprovisionamiento del entorno local comienza con la autenticación del usuario y la habilitación de los servicios de orquestación en la Google Cloud Console o mediante el SDK de gcloud:

1. **Autenticación Inicial:**

2. **Inicialización y APIs:**`gcloud init` (seleccione su proyecto y región `us-central1`).Habilite los servicios: `aiplatform.googleapis.com`, `cloudresourcemanager.googleapis.com` y `iamcredentials.googleapis.com`.

### 2.2. Gestión de Cuentas de Servicio

Se debe instanciar una cuenta de servicio dedicada (`adk-local-sa`) para desacoplar las credenciales personales del entorno de ejecución:

```
gcloud iam service-accounts create adk-local-sa --display-name="Cuenta ADK Local"
gcloud projects add-iam-policy-binding [PROJECT_ID] \
    --member="serviceAccount:adk-local-sa@[PROJECT_ID].iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```
Tras generar la clave JSON mediante `gcloud iam service-accounts keys create ~/adk-key.json`, es obligatorio añadir el archivo al `.gitignore` del repositorio para evitar su exposición accidental en sistemas de control de versiones.

### 2.3. Variables de Entorno y Autenticación

Para que el SDK localice las credenciales de forma transparente, configure la variable `GOOGLE_APPLICATION_CREDENTIALS`. En entornos Colab, se prefiere el método `auth.authenticate_user()` para sesiones efímeras, mientras que en local se utiliza la ruta absoluta al JSON generado.

## 3. Arquitectura y Configuración del Agente de IA

La arquitectura ADK es modular y se fundamenta en la interacción entre el modelo, el planificador (Planner) y las herramientas. Esta estructura permite que el agente no solo genere texto, sino que ejecute acciones lógicas y mantenga un estado persistente a través de ciclos de interacción.

### 3.1. Inicialización del Agente y Planificación

El corazón del sistema es el objeto `aiagent.Agent`. La selección del modelo debe ser balanceada: **Gemini-1.5-Pro** se reserva para tareas de razonamiento complejo y análisis de gran contexto, mientras que **Gemini-1.5-Flash** se emplea para optimizar costos y latencia en tareas directas. Es imperativo integrar un **ReActPlanner**, que dota al agente de la capacidad de "Pensar-Actuar-Observar".

### 3.2. Implementación de Herramientas (Tools) con Lógica Robusta

En lugar de simples wrappers, las herramientas deben incluir validación y manejo de errores. A continuación se presenta una implementación profesional para un agente con herramientas personalizadas:

```
from google.cloud import aiagent
import json
from datetime import datetime

class AgentePersonalizado:
    def _herramienta_calculadora(self, expresion: str) -> str:
        """Evaluación segura de expresiones matemáticas."""
        try:
            allowed_chars = set("0123456789+-*/(). ")
            if all(c in allowed_chars for c in expresion):
                return f"Resultado: {eval(expresion)}"
            return "Error: Caracteres no permitidos."
        except Exception as e:
            return f"Error en cálculo: {str(e)}"

    def _herramienta_fecha(self, formato: str = "%Y-%m-%d") -> str:
        """Retorna la fecha actual del sistema."""
        return datetime.now().strftime(formato)

# Integración en ADK
instancia_herramientas = AgentePersonalizado()
agent = aiagent.Agent(name="agente-ops", model="gemini-1.5-flash")

for name, func in [("calculadora", instancia_herramientas._herramienta_calculadora), 
                   ("obtener_fecha", instancia_herramientas._herramienta_fecha)]:
    tool_wrapper = aiagent.Tool.from_function(
        func=func, name=name, description=f"Herramienta para {name}"
    )
    agent.add_tool(tool_wrapper)
```
### 3.3. Persistencia y Memoria de Conversación

La continuidad se gestiona mediante `ConversationMemory`. El parámetro `max_turns` es crítico para no saturar la ventana de contexto del modelo, mientras que `storage_path` permite la persistencia en disco del historial, vital para la recuperación tras reinicios del servicio.

## 4. Protocolo de Validación Técnica y Pruebas Unitarias

La fiabilidad de un sistema de IA no puede basarse en pruebas manuales. Es necesario implementar una suite de pruebas automatizadas que valide la lógica de los componentes y la recuperación de la memoria.

### 4.1. Suite de Pruebas con Unittest

El siguiente script (`test_agente.py`) es el estándar para validar la integridad del agente antes de cualquier despliegue:

```
import unittest
import os
from mi_agente_local import MiAgenteLocal # Asumiendo clase del paso anterior

class TestAgenteLocal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agente = MiAgenteLocal(model_name="gemini-1.5-flash")

    def test_inicializacion(self):
        """Verifica que el agente y sus componentes existan."""
        self.assertIsNotNone(self.agente.agent)
        self.assertIsNotNone(self.agente.memory)

    def test_memoria_conversacion(self):
        """Valida que el agente recuerde información en turnos sucesivos."""
        self.agente.consultar("Mi nombre es Operador-IA")
        respuesta = self.agente.consultar("¿Cuál es mi nombre?")
        self.assertIn("Operador-IA", respuesta.text)

    def test_guardado_carga(self):
        """Prueba la persistencia del estado en disco."""
        ruta_test = "./test_estado.json"
        self.agente.guardar_estado(ruta_test)
        self.assertTrue(os.path.exists(ruta_test))
        if os.path.exists(ruta_test): os.remove(ruta_test)

if __name__ == "__main__":
    unittest.main(verbosity=2)
```
### 4.2. Monitoreo de Recursos

Durante la ejecución de las pruebas, se debe supervisar el consumo de recursos. En entornos Linux/WSL2, utilice `htop` para RAM y `nvidia-smi -l 1` para observar el consumo de VRAM si se utilizan modelos locales o procesamiento paralelo.

## 5. Parámetros de Despliegue y Configuración de Producción

El paso a producción requiere transformar el código experimental en un servicio robusto, escalable y monitoreable mediante infraestructura gestionada.

### 5.1. Configuración de Producción (YAML)

La configuración se externaliza en archivos YAML para permitir cambios sin modificar el código fuente, definiendo límites de tasa y umbrales de alerta:

```
version: "1.0"
agent:
  name: "agente-produccion"
  model: "gemini-1.5-pro"
  timeout_seconds: 60
  rate_limit:
    requests_per_minute: 100
    requests_per_day: 10000
monitoring:
  enable_logging: true
  metrics: ["latency", "success_rate", "token_usage"]
  alerting:
    thresholds:
      error_rate: 0.05
      p99_latency: 5000 # ms
```
### 5.2. Despliegue en Vertex AI

Utilice el método `agent.deploy()` para instanciar el agente en la infraestructura de Google. Se recomienda el tipo de máquina `n1-standard-4` y configurar el auto-escalado (`min_replica_count=1`, `max_replica_count=3`) para manejar la variabilidad del tráfico sin incurrir en costos excesivos.

### 5.3. Contenerización y Portabilidad

El `Dockerfile` garantiza que el agente se ejecute en un entorno inmutable. Se utiliza una imagen `slim` para minimizar vulnerabilidades y se define un usuario no root por seguridad.

```
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Seguridad: Ejecución como usuario no root
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/key.json
ENV PYTHONPATH=/app
ENV PORT=8080

EXPOSE 8080

CMD ["python", "app/main.py"]
```
## 6. Resolución de Problemas y Mantenimiento Operativo

La estabilidad a largo plazo depende de un marco proactivo de resolución de incidencias.

### 6.1. Matriz de Solución de Problemas

| Problema | Causa Probable | Solución Técnica |
| --- | --- | --- |
| Error de autenticación | Credenciales inválidas o expiradas | Verificar GOOGLE_APPLICATION_CREDENTIALS; regenerar JSON. |
| ImportError: módulo no encontrado | Entorno virtual desactivado | Activar venv; ejecutar pip install -r requirements.txt. |
| Límites de cuota excedidos | Uso excesivo de APIs de Vertex | Solicitar aumento en GCP Console; implementar caché de respuestas. |
| Problemas con WSL2 | Configuración de memoria insuficiente | Ejecutar wsl --update; ajustar RAM en .wslconfig. |
| Falta de memoria | Modelo muy pesado o contexto grande | Migrar a gemini-1.5-flash; reducir max_turns. |

### 6.2. Rutinas de Mantenimiento

1. **Auditoría de Versiones:** Ejecute `pip list --outdated` mensualmente para parchar vulnerabilidades.

2. **Higiene de Secretos:** Revocar y rotar las claves JSON de las cuentas de servicio cada 90 días.

3. **Optimización de Costos:** Analizar regularmente el consumo de tokens en la consola de Google Cloud para ajustar los límites de tasa del agente.

\--------------------------------------------------------------------------------

**LISTA DE VERIFICACIÓN FINAL PARA LANZAMIENTO**

• \[ \] Credenciales JSON excluidas del control de versiones mediante `.gitignore`.

• \[ \] `ReActPlanner` integrado y configurado en el agente.

• \[ \] Suite de pruebas unitarias ejecutada con 100% de éxito.

• \[ \] Archivo `config_produccion.yaml` validado con umbrales de alerta.

• \[ \] Dockerfile configurado con usuario `appuser` (no-root).