# SuperSBT: Faster and Stronger Single Branch Transformer Tracking

## Abstract
Existing transformer-based trackers which are built on two popular tracking paradigms, \ie Siamese-based and DCF-based tracking, mostly leverage transformer model to solely focus on one of these three issues: feature extraction, feature enhancement or feature fusion. Differently, this work presents a novel tracking architecture on top of Single Branch Transformer (SBT). Through two crucial modifications for tracking problem, \ie dynamic feature extraction and correspondence establishment, our conceptually neat tracking framework which is named SuperSBT, simultaneously addresses above three issues. Specifically, with dedicated non-parametric attention operator design, SuperSBT can extract target-dependent features as well as building comprehensive interactions between target and search area, while achieving high inference speed and marvelous tracking performance. We further conduct extensive investigations on the crucial design and architecture variants to provide more insights for SBT tracking. Through our experiments, SuperSBT sets a new record while still running at high inference speed. 

### Code and paper will be released publicily. 

## Results
We obtain the state-of-the-art results on several benchmarks while running at high speed. More results are coming soon. 
<table>
  <tr>
    <th>Model</th>
    <th>GOT-10k<br>AO (%)</th>
    <th>Speed<br></th>
    <th>Params<br></th>
  </tr>
  <tr>
    <td>SuperSBT-tiny</td>
    <td>64.2</td>
    <td>80.9</td>
    <td>70fps</td>
    <td>16.7M</td>
  </tr>
  <tr>
    <td>SuperSBT-small</td>
    <td>64.2</td>
    <td>80.9</td>
    <td>70fps</td>
    <td>16.7M</td>
  </tr>
  <tr>
    <td>SuperSBT-base</td>
    <td>64.2</td>
    <td>80.9</td>
    <td>70fps</td>
    <td>16.7M</td>
  </tr>
  <tr>
