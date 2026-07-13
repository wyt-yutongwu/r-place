# How Millions Coordinate at Scale: Engagement, Collaboration, and Conflict in Three Editions of Reddit r/place 

[![arXiv](https://img.shields.io/badge/arXiv-2408.13236-b31b1b.svg)](https://arxiv.org/abs/2408.13236)

Code and data for the paper **[How Millions Coordinate at Scale: Engagement, Collaboration, and Conflict in Three Editions of Reddit r/place ](https://arxiv.org/abs/2408.13236)** by Yutong Wu and Arlei Silva.

## Overview

The r/place experiment is a series of three social games hosted by Reddit that produce fine-grained traces of the sequential actions taken by millions of players. This work is the first to characterize collective behavior during r/place in terms of **engagement**, **collaboration**, and **conflict** using tools from computational social science and data science.

Our analysis shows that r/place reflected many recurring organizational patterns across editions:

- Participation remains highly concentrated to few participants despite individual rate-limiting constraints
- Larger coalitions exhibit both greater coordination inefficiencies and lower median per-participant activity despite being successful
- Analysis of recovered coalition trajectories further shows that coalition outcomes are difficult to predict based on state characteristics during much of the event, highlighting the dynamic and contested nature of collaborative production in r/place

These findings provide new insight into the lifecycle of large-scale online collaboration and suggest design considerations for collaborative systems that must balance participation, coordination, and competition.

## Repository Structure

```
.
├── r:place/          # Code for Segmentation
├── Collaboration.ipynb           # Analysis code for collaboration
├── Competition.ipynb     # Analysis code for competition
├── Engagement-Copy1.ipynb # Analysis code for engagement
└── README.md
```

## Data

Raw r/place tile placement data is publicly released by Reddit:

- [r/place 2017](https://www.reddit.com/r/redditdata/comments/6640ru/place_datasets_april_fools_2017/)
- [r/place 2022](https://www.reddit.com/r/place/comments/txvk2d/rplace_datasets_april_fools_2022/)
- [r/place 2023](https://www.reddit.com/r/place/comments/15bjm5o/rplace_2023_data/)

## Citation

If you find this work useful, please cite:

```bibtex
@article{wu2024noplace,
  title={No Place for Old Memes: Large-scale Collective Dynamics in the Three Iterations of Reddit r/place},
  author={Wu, Yutong and Silva, Arlei},
  journal={arXiv preprint arXiv:2408.13236},
  year={2024}
}
```

## Contact

Yutong Wu — yw180@rice.edu
