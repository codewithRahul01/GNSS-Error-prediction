GNSS Error Prediction 


The accuracy of GNSS(Gobal Navigation Satellite System) is fundamentally limited by error in:
1. Satellite Clock Biases
2. Satellite ephemeris Prediction

These error, if not accurately modelled can lead to  significant deveation in positioning and timing.

Modern GNSS systems broadcast predicted satellite clock and orbit information to users. However, these broadcast values differ from the actual satellite clock and orbit values due to:

Clock drift
Clock aging
Orbital perturbations
Solar radiation pressure
Gravitational effects
Modeling inaccuracies

As a result, positioning accuracy degrades, especially for high-precision applications such as:

Autonomous vehicles
Aircraft navigation
Surveying
Precision agriculture
Space missions

This challenge aims to leverage Artificial Intelligence (AI) and Machine Learning (ML) techniques to predict future GNSS clock and ephemeris errors, thereby improving navigation accuracy.

Problem Description

Participants will be provided with a dataset containing:

Historical Data

Seven days of recorded:

Satellite clock errors
Satellite ephemeris (orbit) errors

for GNSS satellites operating in:

GEO/GSO Satellites
Geostationary Orbit
Geosynchronous Orbit
MEO Satellites
Medium Earth Orbit

Objective

Develop an AI/ML model capable of learning temporal patterns in GNSS clock and orbit errors and predicting future error values.

The model must predict Clock Error Ephemeris Error for an unseen eighth day.