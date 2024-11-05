# English, Galician and Irish PoS Taggers

This repository contains the implementation of the second assignment of the course of Natural Language Processing of the Master in Artificial Intelligence at the University of Santiago de Compostela. In this assignment, we trained a PoS tagger for English, Galician and Irish languages using the Universal Dependencies corpora.

## Project structure :open_file_folder:

The project is structured as follows:

- `data/`: contains the corpora used for training and testing the taggers.
- `models/`: contains the trained models for each language.
- `gal_tagger.ipynb`: contains the implementation of the Galician PoS tagger and the experiments carried out.
- `irish_tagger.ipynb`: contains the implementation of the Irish PoS tagger and the experiments carried out.
- `tagger.ipynb`: contains the implementation of the English PoS tagger and the experiments carried out.
- `tagger.py`: contains the implementation of the PoS tagger class.
- `utils.py`: contains utility functions used in the notebooks.

In the notebooks explanations of the code and the results of the experiments are provided.

## Requirements :page_with_curl:

With the following command you can install the required packages:

```bash
pip install -r requirements.txt
```

## Datasets :floppy_disk:

The datasets used for training and testing the taggers are the following:

- English: [UD English EWT](https://github.com/UniversalDependencies/UD_English-EWT).
- Galician: [UD Galician-CTG](https://github.com/UniversalDependencies/UD_Galician-CTG)
- Irish: [UD Irish-IDT](https://github.com/UniversalDependencies/UD_Irish-IDT)

All the datasets are available in the `data/` folder. The datasets are already split into training validation and testing sets.

## Usage :writing_hand:

The notebooks are self-contained and can be run independently. Simply running the cells in each notebook will load the pre-trained models stored in the `./models` directory and evaluate them. This avoids the need to retrain the models, which can be time-consuming.

If you want to retrain the models, the necessary code is provided as comments in each cell where the model is loaded. To do this, uncomment the code for training, comment out the code for loading the models, and then run the cells.

## Model training and hyperparameter tuning :microscope:

**Step 1: Training the English Tagger**

Initially, we trained the English tagger and conducted several experiments to identify the optimal hyperparameter configurations. We focused on tuning the following hyperparameters:

- Bidirectional LSTM layers (enabled or disabled).
- LSTM units.
- Embedding dimension.
- Batch size.

**Step 2: Training the Galician Tagger**

After determining the best configuration for English, we used these hyperparameters as a baseline for training the Galician and Irish taggers.

For Galician, we further tuned specific hyperparameters to optimize performance for this language, omitting batch size adjustments since results on the English dataset indicated minimal impact from this parameter. We also chose not to test a model without bidirectional LSTM layers, as we anticipated a decrease in performance. Thus, the hyperparameters adjusted were:

- LSTM units.
- Embedding dimension.

Overall, we observed that changes to these hyperparameters had limited impact on performance. However, to fully verify this, conducting multiple experiments to analyse the variance in loss and accuracy on the test set would be beneficial.

**Step 3: Training the Irish Tagger**

For Irish, we performed minimal tuning, focusing solely on the LSTM units. Similar to Galician, adjustments in hyperparameters showed no significant effect on performance.

## Authors :busts_in_silhouette:

- [Alejandro Esperón Couceiro](https://github.com/Alexec02)
- [Antón Gómez López](https://github.com/antongomez)
- [Eliseo Pita Vilariño](https://github.com/elipitav)

## License :page_facing_up:

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
