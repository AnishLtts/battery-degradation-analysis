Battery Degradation Analysis:

An end-to-end Battery Degradation Analysis platform built python and stream lit to analysis battery 
ageing data, diagnosis degradation mechanism, estimate their contribution, and visualise battery
health across charge-discharge cycles.

Live Demo:

The interactive stream lit dashboard allows users to explore battery degradation results, select
individual batteries, inspect degradatiion diagnosis, confidence scores, and capacity fade across cycles.
live Demo link-- https://battery-degradation-analysis-k6d43mnsr2zhmx5hpvkxfb.streamlit.app/

Project Overview:

Battery Performance gradually deteriorates as a battery undergoes repeated charge-discharge cycles. 
Understanding not only how much degradation has occurred, but also the underlying degradation mechanism, is important for battery health analysis.

this Project develops a data driven battery degradation analysis pipeline using battery ageing data.

The system is divided into two main components: 

MODEL 1: Degradation mechanism Diagnosis 

Model 1 focuses on identifying the dominant degradation mechanism for each analysed battery cycles.

The results are presented through the stream lit dashboard together with a confidence score, allowing degradation behaviour to be inspected across battery cycles. 

MODEL 2: Degradation Contribution Analysis

Model 2 focuses on analysis how different degradation mechanism contribute to overall battery degradation 

This complements Model 1 by moving form:

"Which degradation mechanism is occurring?"

to

"How much does each mechanism contribute?"


