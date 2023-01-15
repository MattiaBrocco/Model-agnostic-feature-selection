# CBSD-2022-UNIPD
This study focuses on the replicability of finding relevant predictors for lie detection in various psychometric tests concerning medicine, behavioral science and data science that have been compiled twice, once honestly and once dishonestly. More precisely, the goal is to develop a framework for feature selection that leads to good and similar results for different models used for the discrimination of honesty and dishonesty of test responses. Accuracy, Top-5 stability and Accuracy Standard Deviation are the metrics used to evaluate the results.

![Overall_comparison](https://user-images.githubusercontent.com/61026948/212570001-d7963dfc-85a7-4cf2-8304-742f7a3f1685.jpg)

## *Approaches used*
The approaches developed in this project to select the features are the following: 
  1. **PCA**: select 20% of the total number of features using the principal component analysis.
  2. **Permutation importance**: fitted on a random forest, with features selected based on t-test.
  3. **Mutual Information**: the features selected by the Joint Mutual Information Maximization (JMIM) algorithm with an importance score of at least 0.8 out of 1 are used.

Before applying the methods, the datasets are split into training and test (70%-30%) and for every feature, the mean and the standard deviation are computed in order to scale that feature: $Z=\frac{X-\mu}{\sigma}$. 
These three methods are independent of each other and each one of them is going to be described in depth later on. 

## *Models used*
Each one of the approaches considered in this project, as mentioned before, selects a number of features from the corresponding original dataset, and then these selected features are used to train different models and to observe their performance. The models trained in this project are: 
  1. Logistic regression model on all the features (Full LR)
  2. Logistic regression model on selected features (LR)
  3. Support vector machine (SVM)
  4. Random forest (RF)
  5. Multi-layer perceptron classifier (MLP) 

For each of these models is also computed the related accuracy in order to see firstly how good that model is performing with the selected features and secondly to compare the models between them in order to figure out if the selected features give similar performances among all the models. 
A logistic regression with all the features is trained at the beginning. In this way, it’s possible to have a comparison between the results obtained with the selected features.

## *Metrics*

* Accuracy: ratio of correct predictions over the number of instances. This has been chosen as all the datasets show a fairly balanced number of examples per class (all are binary classification tasks). The accuracy is computed on the full model (Full LR) as well as all the other four models used for benchmarking and trained only on the subset of features selected by each of the procedures in scope.
* Accuracy Standard Deviation: standard deviation of the four models (i.e. LR, SVM, RF, MLP) fitted on the subset of selected features. It is a measure of the consistency of the classification performance across different models, thus the lower the better.
* Top-5 stability: a more specific metric for assessing consistency across models (i.e. LR, SVM, RF, MLP). It takes into account the first five most important features used by each of the models, the formula developed is:  

<center>
    $TOP5 Stability=1-\big(\frac{1}{(\# models-1)\cdot\min{(5,|\Omega|)}} \sum^{\min{(5,|\Omega|)}}_{i=1}{|\beta_{i}|-1}\big)$
</center>
where $\Omega$ is the set of features selected by a procedure, i.e. $\Omega=\{\beta_1,...,\beta_n\}$; $\beta_i$ is a vector with the feature selected with importance $i$ across the models (notice that in our case $\# models=4$). Finally, $|\beta_i|$ is the number of unique values in $\beta_i$.</p>

### *Datasets*

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Name </th>
    <th class="tg-0pky">Topic</th>
    <th class="tg-0pky">Faking good/faking bad</th>
    <th class="tg-0pky">Number of samples</th>
    <th class="tg-0pky">Numbers of features</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">DT_df_CC</td>
    <td class="tg-0pky">Short Dark Triad 3 for child costudy</td>
    <td class="tg-0pky">Faking good </td>
    <td class="tg-0pky">482</td>
    <td class="tg-0pky">27</td>
  </tr>
  <tr>
    <td class="tg-0pky">DT_df_JI</td>
    <td class="tg-0pky">Short Dark Triad 3  for a job interview </td>
    <td class="tg-0pky">Faking good</td>
    <td class="tg-0pky">864</td>
    <td class="tg-0pky">27</td>
  </tr>
  <tr>
    <td class="tg-0pky">PRMQ_df</td>
    <td class="tg-0pky">Identify memory difficulties</td>
    <td class="tg-0pky">Faking bad </td>
    <td class="tg-0pky">1404</td>
    <td class="tg-0pky">16</td>
  </tr>
  <tr>
    <td class="tg-0pky">PCL5_df</td>
    <td class="tg-0pky">Identify victims of PTSD</td>
    <td class="tg-0pky">Faking bad</td>
    <td class="tg-0pky">402</td>
    <td class="tg-0pky">20</td>
  </tr>
  <tr>
    <td class="tg-0pky">NAQ_R_df </td>
    <td class="tg-0pky">Identify possible victims of mobbing</td>
    <td class="tg-0pky">Faking bad</td>
    <td class="tg-0pky">712</td>
    <td class="tg-0pky">22</td>
  </tr>
  <tr>
    <td class="tg-0pky">PHQ9_GAD7_df</td>
    <td class="tg-0pky">Identify possible victims of anxious-depressive syndrom</td>
    <td class="tg-0pky">Faking bad</td>
    <td class="tg-0pky">1118</td>
    <td class="tg-0pky">16</td>
  </tr>
  <tr>
    <td class="tg-0pky">PID5_df</td>
    <td class="tg-0pky">Identify mental disorders</td>
    <td class="tg-0pky">Faking bad</td>
    <td class="tg-0pky">824</td>
    <td class="tg-0pky">220</td>
  </tr>
  <tr>
    <td class="tg-0pky">sPID5_df</td>
    <td class="tg-0pky">Identify mental disorders </td>
    <td class="tg-0pky">Faking bad</td>
    <td class="tg-0pky">1038</td>
    <td class="tg-0pky">25</td>
  </tr>
  <tr>
    <td class="tg-0pky">PRFQ_df</td>
    <td class="tg-0pky">Specific caregivers' ability to mentalize with their children</td>
    <td class="tg-0pky">Faking good</td>
    <td class="tg-0pky">678</td>
    <td class="tg-0pky">18</td>
  </tr>
  <tr>
    <td class="tg-0pky">IESR_df</td>
    <td class="tg-0pky">Identify possible victims of PTSD</td>
    <td class="tg-0pky">Faking bad</td>
    <td class="tg-0pky">358</td>
    <td class="tg-0pky">22</td>
  </tr>
  <tr>
    <td class="tg-0pky">R_NEO_PI_df</td>
    <td class="tg-0pky">Personality questionnaire (Big5)</td>
    <td class="tg-0pky">Faking good</td>
    <td class="tg-0pky">77687</td>
    <td class="tg-0pky">30</td>
  </tr>
  <tr>
    <td class="tg-0pky">RAW_DDDT_df</td>
    <td class="tg-0pky">Identify Dark Triad personality</td>
    <td class="tg-0pky">Faking bad</td>
    <td class="tg-0pky">986</td>
    <td class="tg-0pky">12</td>
  </tr>
  <tr>
    <td class="tg-0pky">IADQ_df</td>
    <td class="tg-0pky">Identify adjustment disorder (stress response syndrome)</td>
    <td class="tg-0pky">Faking bad</td>
    <td class="tg-0pky">450</td>
    <td class="tg-0pky">9</td>
  </tr>
  <tr>
    <td class="tg-0pky">BF_df_CTU</td>
    <td class="tg-0pky">Job interview for a salesperson position</td>
    <td class="tg-0pky">Faking good</td>
    <td class="tg-0pky">442</td>
    <td class="tg-0pky">10</td>
  </tr>
  <tr>
    <td class="tg-0pky">BF_df_OU</td>
    <td class="tg-0pky">Job interview for in humanitarian organization</td>
    <td class="tg-0pky">Faking good</td>
    <td class="tg-0pky">460</td>
    <td class="tg-0pky">10</td>
  </tr>
  <tr>
    <td class="tg-0pky">BF_df_V</td>
    <td class="tg-0pky">Obtain child costudy </td>
    <td class="tg-0pky">Faking good</td>
    <td class="tg-0pky">486</td>
    <td class="tg-0pky">10</td>
  </tr>
</tbody>
</table>



