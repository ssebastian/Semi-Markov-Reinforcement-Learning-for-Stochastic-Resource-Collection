# Semi-Markov Reinforcement Learning for Stochastic Resource Collection


## Abstract
We show that the task of collecting stochastic, spatially distributed resources (Stochastic Resource Collection, SRC) may be considered as a Semi-Markov-Decision-Process. Our Deep-Q-Network (DQN) based approach uses a novel scalable and transferable artificial neural network architecture. The concrete use-case of the SRC is an officer (single agent) trying to maximize the amount of fined parking violations in his area. We evaluate our approach on a environment based on the real-world parking data of the city of Melbourne. In small, hence simple, settings with short distances between resources and few simultaneous violations, our approach is comparable to previous work. When the size of the network grows (and hence the amount of resources) our solution significantly outperforms preceding methods. Moreover, applying a trained agent to a non-overlapping new area outperforms existing approaches. 

### Conference Presentation
[Short Video](https://www.ijcai.org/proceedings/2020/video/25156)

[Long Video](https://www.ijcai.org/proceedings/2020/video/26129)

## Install (main system)
Install requirements (consider creating a virtual environment):
```bash
pip install -r requirements.txt
```
You might also need to install following libraries (Ubuntu):
```bash
sudo apt-get install libspatialindex-dev libgdal-dev ffmpeg
```
This is tested on Ubuntu 20.04.2 LTS with Python 3.8.5.


## Create Database
Run:
```bash
python create_db.py
```

## Run
Run:
```bash
python -O main.py
```

## Install and run with docker (no GPU)
Create the docker container:
```bash
cd /path/to/project
docker build -f dockerfile -t toprl .
```

Run the project with:
```bash
cd /path/to/project
docker run --rm -it -v `pwd`:/mnt toprl
```

## Visualize Results
In the `logs/[run_name]/` folder, you will find the trained parameters, results and videos of the agent at given checkpoints.

## Try different parameters
You may want to modify the hyper-parameters at the bottom of the main.py file.

## Competitors
You can start competitors by calling the competitor.py file:
```bash
python -O competitor.py
```

## Citation
If you use this project, please cite:
```bibtex
@inproceedings{ijcai2020-463,
  title     = {Semi-Markov Reinforcement Learning for Stochastic Resource Collection},
  author    = {Schmoll, Sebastian and Schubert, Matthias},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Christian Bessiere},
  pages     = {3349--3355},
  year      = {2020},
  month     = {7},
  note      = {Main track}
  doi       = {10.24963/ijcai.2020/463},
  url       = {https://doi.org/10.24963/ijcai.2020/463},
}
```

Authors: [Sebastian Schmoll](https://www.dbs.ifi.lmu.de/cms/personen/mitarbeiter/schmoll/index.html) and [Matthias Schubert](https://www.dbs.ifi.lmu.de/cms/personen/professoren/schubert/index.html)

See also: [AI-beyond](https://www.ai-beyond.org/)
