# No Place for Old Memes: Large-scale Collective Dynamics in the Three Iterations of Reddit r/place

[![arXiv](https://img.shields.io/badge/arXiv-2408.13236-b31b1b.svg)](https://arxiv.org/abs/2408.13236)

Code and data for the paper **[No Place for Old Memes: Large-scale Collective Dynamics in the Three Iterations of Reddit r/place](https://arxiv.org/abs/2408.13236)** by Yutong Wu and Arlei Silva.

## Overview

The r/place experiment is a series of three social games hosted by Reddit, producing fine-grained traces of sequential actions taken by millions of players. This work is the first to characterize collective behavior during r/place in terms of **engagement**, **collaboration**, and **competition** using tools from computational social science and data science.

Our analysis shows that r/place reflected many patterns found in other group decision-making processes, including empirical evidence for:

- Group coordination costs
- Social loafing
- Increased cooperation as a response to competition

These findings can support the development of new theoretical models, tools, and mechanisms to optimize collaborative-competitive processes in social networks.

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
