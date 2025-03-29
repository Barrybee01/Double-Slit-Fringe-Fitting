This code is merged together from a Google Colab Python notebook. The code can run with all the cells merged with no issue. However, the individual cells are highlighted in the Python file

This code accepts CSV files that contain information about Intensity vs Position for double slit interference. For this case, we used Intensity vs Pixel Number. This is equivalent as pixels are discrete and have a known size.

Experimental parameters like source energy, slit parameters, and source-to-slit distance need to be added manually before running code

Included within this code are calculations for goodness of fit and fringe visibility. The goodness of fit calculation predicts the most likely extension number, while the visibility calculation determines the fringe visibility.

When using this code, run the goodness of fit calculation before adjusting the extension number. Use the output of the goodness of fit as the initial input for the curve fitting

This code was used in this article describing the limitations of coherence length measurements using a double slit apparatus https://arxiv.org/abs/2410.13172
