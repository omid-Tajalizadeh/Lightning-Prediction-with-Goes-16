# FlaSH: Lightning-Prediction-with-Goes-16
## Abstract 

Lightning is a natural weather hazard still claiming thousands of lives globally each year.
Prevention of fatalities due to lightning via rules and regulation is common practice in society but comes with costs and delays. Skillful prediction of the occurrence of lightning is
therefore an important aspect of weather forecasting. Traditionally, lightning occurrence
is predicted by the use of physics-based numerical weather prediction models. However,
such model simulations are computationally expensive and tend to not incorporate the
latest observational information into the prediction. Here, we follow a different approach
by exploring the potential of the use of learning algorithms applied to satellite observations
of clouds over the Caribbean region for providing skillful short-term lightning predictions
for the following 35 minutes (so-called “nowcasting”). Lightning is associated with a particular type of weather and clouds (thunderstorms). Geostationary satellites nowadays
provide continuously updated near-real-time high-quality cloud information at high spatial resolutions. Hence, the possibility exists that satellite observations contain valuable
information about potential near-future lightning occurrences.We train well-known models like ResNet50 and MobileNetV3 as well as a basic convolutional neural network (CNN)
individually on 100,000 patches with an 8km by 8km resolution, taken from the original
satellite images over the entire BES islands (Bonaire, Saint Eustatius, and Saba) region,
to achieve 0.99 accuracy and 0.99 AUC score. By performing a convolution-like operation
on patches of the entire area satellite picture, we introduce a novel approach called FlaSH
(Flash Score Heatmap). For the entire satellite picture, FlaSH achieves an AUC score of
0.96, demonstrating its ability to precisely nowcast lightning.
