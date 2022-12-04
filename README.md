# EdgeRelaxationGraphDrawing
Primary repository for the research project on edge selection for relaxation and graph drawing

## Installation
1. Create python virtual environment (Python 3.10 recommended)
```{bash}
python3 -m venv venv
```
2. Modify activate file to correct PYTHONPATH
```{bash}
echo -e "\n\nexport PYTHONPATH=\"$(pwd)\"" >> venv/bin/activate
```
3. Activate virtual environment
```{bash}
source venv/bin/activate
```
4. Install required libraries
```{bash}
pip3 install -r requirements.txt
```

Then, remember to always activate and deactivate the virtual environment when using the repository:
```{bash}
# Activate virtual environment
source venv/bin/activate

# Do some stuff...

# Deactivate virtual environment
deactivate
```
## 💡 Idea 
Look at the effect that relaxing an edge has on graph beauty metrics (number of crossings, average length of edges, ...) in terms of different characteristics of that edge as well as on the perturbation of the drawing/resulting objective function.

## ⏱️ To do
- Check feature normalization functions
- Generate data frame with all features (normalized and not normalized) with column for graph type
- Train Statistical models and trees (bucketizing), also DL both per graph type and in full
- From last step generate conclusions and write them up

## 👥 Contributors 
- Jordi Cortadella
- Raúl Higueras
- Nathaniel Mitrani
