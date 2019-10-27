# Maxgap Bandit: Adaptive Algorithms for Approximate Ranking
This repository contains the code for algorithms and experiments in the NeurIPS 2019 paper - Maxgap Bandit: Adaptive Algorithms for Approximate Ranking. The paper can be found at https://arxiv.org/abs/1906.00547.

To generate plots for stopping times, run the following commands:
- cd stopping_times
- python generate_data.py
- python stopping_time_plot.py

To generate plots for simulated data, run the following commands:
 - cd simulated_data
 - python generate_data.py
 - python pulls_plot.py
 - python animation_movie.py
 - python mistake_probability_plot.py

To generate plots for Streetview data, run the following commands:
 - cd streetview_data
 - python generate_data.py (takes almost a day to run)
 - python mistake_probability_plot.py
