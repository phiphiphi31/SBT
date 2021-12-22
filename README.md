# SuperSBT: Faster and Stronger Single Branch Transformer Tracking

### We hope this work can attract more researchers/engineers to build a simple but strong Single Branch Transformer (SBT) driven tracker.

### Code and paper will be released publicily. 

## Contacts
* Fei Xie, School of Automation, Southeast University, China, 372998044@qq.com, wechat: 372998044


## Abstract
Existing transformer-based trackers which are built on two popular tracking paradigms, \ie Siamese-based and DCF-based tracking, mostly leverage transformer model to solely focus on one of these three issues: feature extraction, feature enhancement or feature fusion. Differently, this work presents a novel tracking architecture on top of Single Branch Transformer (SBT). Through two crucial modifications for tracking problem, \ie dynamic feature extraction and correspondence establishment, our conceptually neat tracking framework which is named SuperSBT, simultaneously addresses above three issues. Specifically, with dedicated non-parametric attention operator design, SuperSBT can extract target-dependent features as well as building comprehensive interactions between target and search area, while achieving high inference speed and marvelous tracking performance. We further conduct extensive investigations on the crucial design and architecture variants to provide more insights for SBT tracking. Through our experiments, SuperSBT sets a new record while still running at high inference speed.



## Results
We obtain the state-of-the-art results on several benchmarks while running at high speed. More results are coming soon. 
<table>
  <tr>
    <th>Model</th>
    <th>GOT-10k<br>AO (%)</th>
    <th>GOT-10k<br>SR0.5 (%)</th>
    <th>GOT-10k<br>SR0.75 (%)</th>
    <th>Speed<br></th>
    <th>Params<br></th>
  </tr>
  <tr>
    <td>SuperSBT-tiny</td>
    <td>62.6</td>
    <td>73.9</td>
    <td>51.1</td>
    <td>150fps</td>
    <td>10.7M</td>
  </tr>
  <tr>
    <td>SuperSBT-small</td>
    <td>67.6</td>
    <td>77.8</td>
        <td>62.2</td>
    <td>87fps</td>
    <td>25.8M</td>
  </tr>
  <tr>
    <td>SuperSBT-base</td>
    <td>70.0</td>
    <td>80.1</td>
    <td>65.4</td>
    <td>50fps</td>
    <td>52.1M</td>
  </tr>
  <tr>

    
