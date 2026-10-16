# Object-Oriented Programming (OOP) Mini-Projects
## Overview

This repository contains five Python mini-projects developed to apply OOP to practical problems. The projects use examples relevant to Uganda and demonstrate how Python classes, numerical methods, data analysis, forecasting, simulation, and automated testing can support problem-solving.

The projects cover population forecasting, solar micro-grid dispatch, fish stock risk, rainfall-based crop suitability, and taxi route revenue and fleet planning.

## Mini-Projects

### Project 1: UBOS District Population Forecaster

**Notebook:** `project1_population.ipynb`

This project explores population forecasting using district-level population data. It implements a `DistrictPopulation` class to organise population observations by district and year, validate the input data, and support forecasting analysis.

The example data cover Kampala, Wakiso, Gulu, Mbarara, and Mbale. The analysis explores linear trends, exponential growth, Fibonacci-ratio-based modelling, five-year forecasts, and uncertainty assessment through bootstrapping.

**Main methods and concepts:**
- Object-oriented data representation and input validation.
- Population trend analysis and forecasting.
- Comparison of alternative forecasting approaches.
- Forecast uncertainty and limitations of projecting historical trends into the future.

**Source and tests:**
- `src/population_forecasting.py`
- `tests/test_population_forecasting.py`

### Project 2: Solar Micro-Grid Dispatch Planner

**Notebook:** `project2_solar_microgrid/project2_solar_microgrid.ipynb`

This project investigates how a rural health centre in Kasese can meet its energy demand using solar power and battery storage. It formulates energy allocation as a system of linear equations and uses SciPy to solve the system. The project also explores non-negative least squares (NNLS), a hybrid micro-grid model that includes diesel generation, and Monte Carlo simulation to assess the feasibility and variability of energy requirements.

**Main methods and concepts:**
- Linear algebra and solving simultaneous equations.
- Feasibility checks for solar and battery allocations.
- Non-negative least squares for constrained solutions.
- Hybrid energy-system modelling and Monte Carlo simulation.
- Synthetic 30-day demand data with a weekly pattern and random variation.

**Selected findings:** The base model's coefficient matrix has determinant -5 and condition number approximately 5.83. In the reported 1,000-run hybrid-system simulation, no simulated cases were classified as infeasible. These findings apply to the model assumptions and synthetic data, rather than measured performance from an operating health centre.

**Source, data, and tests:**
- `project2_solar_microgrid/src/microgrid.py`
- `project2_solar_microgrid/tests/test_microgrid.py`
- `project2_solar_microgrid/demands_30_days.csv`

### Project 3: Fish Stock Risk

**Notebook:** `project3_fish_stock_risk/project3_fish_stock.ipynb`

This project applies object-oriented programming to a fish-stock risk problem. Its implementation and analysis are documented in the notebook and supporting Python module.

**Source and tests:**
- `project3_fish_stock_risk/src/fish_stock.py`
- `project3_fish_stock_risk/tests/test_fish_stock.py`

### Project 4: Rainfall Crop Suitability

**Notebook:** `project4_rainfall_crop_suitability/project4_rainfall_crop_suitability.ipynb`

This project explores rainfall-related crop suitability using Python-based data analysis. The notebook contains the analysis and supporting methods, while the source module and automated tests provide reusable implementation and validation.

**Source and tests:**
- `project4_rainfall_crop_suitability/src/rainfall.py`
- `project4_rainfall_crop_suitability/tests/test_rainfall.py`

### Project 5: Taxi Route Revenue, Pricing, and Fleet Planner

**Notebook:** `project5_taxi_fleet_planner/project5_taxi_fleet_planner.ipynb`

This project analyses passenger demand, revenue, pricing, forecasting, and vehicle requirements across three illustrative taxi routes: Kampala–Ntinda, Kampala–Entebbe, and Kampala–Mukono.

The project uses a `Route` class to organise route-level passenger and revenue information. It also implements moving-average, exponential-smoothing, and linear-trend forecasting approaches, compares forecasting performance using walk-forward validation, and explores demand patterns using simulated data.

**Selected findings:**
- The observed sample contains 474 passenger trips on the Kampala–Ntinda route, 678 on Kampala–Entebbe, and 520 on Kampala–Mukono. Based on the illustrative fares used in the analysis, the corresponding total revenues are UGX 948,000, UGX 3,390,000, and UGX 1,560,000.
- Exponential smoothing with alpha = 1.0 achieved the lowest reported walk-forward mean absolute error among the evaluated methods for all three routes. With this parameter, the forecast is equivalent to using the most recent observation.
- The modelled supply-demand equilibrium is a fare of UGX 2,200 and a quantity of 76 passengers per trip-hour. This is a theoretical result based on the specified demand and supply equations.
- Under the stated assumptions of 14 passengers per vehicle, eight one-way trips per day, and a 15% demand buffer, the fleet-planning calculation recommends one vehicle per route, or three vehicles in total.

The passenger counts, fares, and simulations are illustrative. The equilibrium quantity is measured per trip-hour, whereas the route passenger counts are daily observations, so the measures should not be directly compared.

**Source and tests:**
- `project5_taxi_fleet_planner/src/taxi_route.py`
- `project5_taxi_fleet_planner/src/forecasters.py`
- `project5_taxi_fleet_planner/tests/test_taxi_route.py`
- `project5_taxi_fleet_planner/tests/test_forecasters.py`

## Repository Structure

```text
Assignment1_MiniProjects/
├── README.md
├── requirements.txt
├── project1_population.ipynb
├── src/
│   └── population_forecasting.py
├── tests/
│   └── test_population_forecasting.py
├── project2_solar_microgrid/
│   ├── project2_solar_microgrid.ipynb
│   ├── demands_30_days.csv
│   ├── src/
│   │   └── microgrid.py
│   └── tests/
│       └── test_microgrid.py
├── project3_fish_stock_risk/
│   ├── project3_fish_stock.ipynb
│   ├── src/
│   │   └── fish_stock.py
│   └── tests/
│       └── test_fish_stock.py
├── project4_rainfall_crop_suitability/
│   ├── project4_rainfall_crop_suitability.ipynb
│   ├── src/
│   │   └── rainfall.py
│   └── tests/
│       └── test_rainfall.py
└── project5_taxi_fleet_planner/
    ├── project5_taxi_fleet_planner.ipynb
    ├── src/
    │   ├── forecasters.py
    │   └── taxi_route.py
    └── tests/
        ├── test_forecasters.py
        └── test_taxi_route.py
```

## Technologies and Dependencies

The projects use Python and selected scientific-computing and testing libraries:

- **NumPy:** numerical computation and simulation.
- **pandas:** data manipulation and analysis.
- **Matplotlib:** data visualisation.
- **SciPy:** linear algebra, constrained optimisation, and scientific calculations.
- **Requests:** HTTP requests where required by a project.
- **pytest:** automated testing.

The shared dependencies are listed in `requirements.txt`.

## Installation and Setup

Use Python 3.11 or a compatible Python version supported by the installed dependencies.

## Running the Notebooks

Open the repository folder in Visual Studio Code and select a Python kernel with the required dependencies installed.

Open the notebook for the project you want to run. Restart the kernel and run all cells from top to bottom to verify that the notebook executes independently.

For Project 2, ensure `demands_30_days.csv` remains in the same project folder as the notebook because the analysis uses that file.

## Running the Automated Tests

Run tests from each project's directory so that its source modules can be imported using the expected project structure.

For Project 1, run from the repository root:

```powershell
python -m pytest tests/ -v
```

For Projects 2–5, first change into the relevant project directory. For example:

```powershell
cd project2_solar_microgrid
python -m pytest tests/ -v
```

Use the corresponding folder names for the remaining projects:

```text
project3_fish_stock_risk
project4_rainfall_crop_suitability
project5_taxi_fleet_planner
```

Each project has its own test suite. Confirm that all tests pass before submitting the repository.

## Limitations

The results depend on the quality of the input data, model assumptions, and methods selected. Synthetic data and illustrative prices or demand values are useful for demonstrating analytical methods but should not be treated as verified real-world measurements.

Forecasts based on short historical series may not capture structural changes, unusual events, or longer-term trends. Energy feasibility and fleet recommendations depend on the constraints specified in the models. Findings should therefore be interpreted in the context of each project's assumptions and limitations.

## Reproducibility and Submission
The notebooks, source modules, and tests are organised within a single repository. Random simulations should use explicitly seeded random-number generators where applicable to make results reproducible.

