# Software Journey: Voice Agent Async

## Artefactos Generados Durante el Proyecto

Además del código fuente, el proyecto produjo múltiples artefactos de ingeniería utilizados para apoyar la toma de decisiones arquitectónicas:

- GitHub Issues
- QA Seam Reviews
- Handoffs entre sesiones
- Architecture Checkpoint
- Auditorías de consistencia
- Suite automatizada de pruebas

Estos artefactos fueron utilizados como evidencia para construir el presente Software Journey.

## Introducción

Este documento presenta el recorrido completo de diseño, construcción y validación del proyecto **Voice Agent Async**, desarrollado durante el semestre utilizando una metodología de colaboración humano-IA basada en agentes y siguiendo el flujo de trabajo descrito en *Running Your AFK Agent*.

El objetivo del proyecto fue construir un agente de voz conversacional mínimo en Python utilizando programación asíncrona (`asyncio`) para comprender en profundidad la arquitectura utilizada por frameworks modernos como Pipecat y LiveKit Agents.

La solución implementa un flujo de procesamiento en tiempo real compuesto por:

```text
Microphone → Speech-to-Text → LLM → Text-to-Speech → Speaker
```

Durante el desarrollo se aplicó un proceso incremental basado en GitHub Issues, revisiones arquitectónicas, auditorías de calidad (QA Seam Reviews), handoffs entre sesiones de trabajo y análisis de diseño inspirados en los principios del libro *A Philosophy of Software Design* de John Ousterhout.

## Estructura del Software Journey

### 1. La Bala Trazadora (Tracer Bullet)

Análisis de la estrategia inicial de reducción de riesgo técnico, la exploración del espacio de diseño y la forma en que los primeros issues permitieron validar tempranamente la arquitectura propuesta.

**Archivo:** `tracer-bullet.md`

### 2. Anatomía de la Complejidad

Evaluación crítica de la calidad del diseño utilizando los conceptos de:

* Deep Modules
* Shallow Modules
* Information Leakage

Se analizan ejemplos concretos del código generado y refinado durante el proyecto.

**Archivo:** `complexity-analysis.md`

### 3. Veredicto Retrospectivo

Reflexión arquitectónica basada en los resultados del Architecture Checkpoint y las decisiones tomadas durante la segunda mitad del desarrollo.

Se evalúan aspectos como:

* Elasticidad de la arquitectura
* Change Amplification
* Buen gusto arquitectónico
* Impacto de los análisis paralelos de diseño

**Archivo:** `retrospective-verdict.md`

## Repositorio

El proyecto se desarrolló mediante un flujo incremental basado en Issues, pruebas automatizadas y revisiones continuas de arquitectura.

Al finalizar el desarrollo se obtuvo:

* Pipeline completamente integrado.
* Suite de pruebas automatizadas pasando al 100%.
* Arquitectura documentada y auditada.
* Integración funcional de todos los componentes del sistema.

## Navegación

- [Sección 1 - Tracer Bullet](tracer-bullet.md)
- [Sección 2 - Anatomía de la Complejidad](complexity-analysis.md)
- [Sección 3 - Veredicto Retrospectivo](retrospective-verdict.md)