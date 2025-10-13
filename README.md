# Advanced Statistical Analysis Toolkit

![R](https://img.shields.io/badge/R-4.0%2B-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Statistics](https://img.shields.io/badge/Statistics-Advanced-purple)

[English](#english) | [Português](#português)

---

<a name="english"></a>
## 🇬🇧 English

### 📊 Overview

**Advanced Statistical Analysis Toolkit** is a comprehensive statistical analysis framework combining the power of **R** for classical statistical methods with **Python** for modern machine learning and visualization. This toolkit provides production-ready implementations of advanced statistical techniques including hypothesis testing, multivariate analysis, bootstrapping, Monte Carlo simulations, and interactive dashboards.

### ✨ Key Features

- **Dual-Language Integration**
  - R for statistical rigor and specialized packages
  - Python for ML, visualization, and workflow automation
  - Seamless integration using rpy2

- **Comprehensive Statistical Methods**
  - Hypothesis testing (t-tests, ANOVA, chi-square, etc.)
  - Multiple regression (linear, logistic, polynomial)
  - Multivariate analysis (PCA, Factor Analysis, Cluster Analysis)
  - Time series analysis (ARIMA, seasonal decomposition)
  - Bayesian inference
  - Survival analysis

- **Resampling Techniques**
  - Bootstrap confidence intervals
  - Permutation tests
  - Cross-validation
  - Monte Carlo simulations

- **Interactive Dashboards**
  - Shiny apps (R)
  - Streamlit dashboards (Python)
  - Plotly visualizations
  - Real-time statistical exploration

### 🏗️ Architecture

```
advanced-statistical-analysis-toolkit/
├── R/                          # R statistical modules
│   ├── hypothesis_testing.R
│   ├── regression_analysis.R
│   ├── multivariate_analysis.R
│   └── time_series_analysis.R
├── python/                     # Python modules
│   ├── statistical_ml.py
│   ├── visualizations.py
│   └── r_integration.py
├── examples/                   # Usage examples
├── tests/                      # Unit tests
├── data/                       # Sample datasets
└── docs/                       # Documentation
```

### 🚀 Quick Start

#### Installation

```bash
# Install R packages
R -e "install.packages(c('tidyverse', 'caret', 'shiny', 'ggplot2', 'stats', 'MASS'))"

# Install Python packages
pip install -r requirements.txt
```

#### Usage Example

```python
from python.r_integration import RStatisticalAnalyzer
import pandas as pd

# Initialize analyzer
analyzer = RStatisticalAnalyzer()

# Load data
data = pd.read_csv('data/sample_data.csv')

# Perform t-test
result = analyzer.t_test(data['group1'], data['group2'])
print(f"p-value: {result['p_value']}")

# Multiple regression
model = analyzer.multiple_regression(
    data=data,
    formula='y ~ x1 + x2 + x3'
)
print(model.summary())
```

### 📄 License

MIT License - see LICENSE file for details.

### 👤 Author

**Gabriel Demetrios Lafis**

---

<a name="português"></a>
## 🇧🇷 Português

### 📊 Visão Geral

**Advanced Statistical Analysis Toolkit** é um framework abrangente de análise estatística que combina o poder do **R** para métodos estatísticos clássicos com **Python** para machine learning moderno e visualização. Este toolkit fornece implementações prontas para produção de técnicas estatísticas avançadas incluindo testes de hipóteses, análise multivariada, bootstrapping, simulações Monte Carlo e dashboards interativos.

### ✨ Principais Recursos

- **Integração Dual-Language**
  - R para rigor estatístico e pacotes especializados
  - Python para ML, visualização e automação de workflows
  - Integração perfeita usando rpy2

- **Métodos Estatísticos Abrangentes**
  - Testes de hipóteses (t-tests, ANOVA, qui-quadrado, etc.)
  - Regressão múltipla (linear, logística, polinomial)
  - Análise multivariada (PCA, Análise Fatorial, Análise de Cluster)
  - Análise de séries temporais (ARIMA, decomposição sazonal)
  - Inferência Bayesiana
  - Análise de sobrevivência

- **Técnicas de Reamostragem**
  - Intervalos de confiança por bootstrap
  - Testes de permutação
  - Validação cruzada
  - Simulações Monte Carlo

- **Dashboards Interativos**
  - Apps Shiny (R)
  - Dashboards Streamlit (Python)
  - Visualizações Plotly
  - Exploração estatística em tempo real

### 🏗️ Arquitetura

```
advanced-statistical-analysis-toolkit/
├── R/                          # Módulos estatísticos R
│   ├── hypothesis_testing.R
│   ├── regression_analysis.R
│   ├── multivariate_analysis.R
│   └── time_series_analysis.R
├── python/                     # Módulos Python
│   ├── statistical_ml.py
│   ├── visualizations.py
│   └── r_integration.py
├── examples/                   # Exemplos de uso
├── tests/                      # Testes unitários
├── data/                       # Datasets de exemplo
└── docs/                       # Documentação
```

### 🚀 Início Rápido

#### Instalação

```bash
# Instale pacotes R
R -e "install.packages(c('tidyverse', 'caret', 'shiny', 'ggplot2', 'stats', 'MASS'))"

# Instale pacotes Python
pip install -r requirements.txt
```

#### Exemplo de Uso

```python
from python.r_integration import RStatisticalAnalyzer
import pandas as pd

# Inicialize o analisador
analyzer = RStatisticalAnalyzer()

# Carregue os dados
data = pd.read_csv('data/sample_data.csv')

# Execute teste t
result = analyzer.t_test(data['group1'], data['group2'])
print(f"p-value: {result['p_value']}")

# Regressão múltipla
model = analyzer.multiple_regression(
    data=data,
    formula='y ~ x1 + x2 + x3'
)
print(model.summary())
```

### 📄 Licença

Licença MIT - veja o arquivo LICENSE para detalhes.

### 👤 Autor

**Gabriel Demetrios Lafis**

