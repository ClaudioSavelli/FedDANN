# FedDANN: Non-generative adversarial learning for federated settings

Claudio Savelli 
Bruno Spaccavento 
Florentin Udrea


--------------------------------------------------------------------------------

The main issue arising when working in the Federated Learning (FL) setting is distributions shift among different clients. In the following work, we analyse the FL framework and a few of the solutions proposed in the literature for its intrinsic problems. 

The base FedAvg algorithm is proposed and analysed in the initial part of the paper in order to have a baseline on the FEMNIST dataset and compare it to the centralised setting. The hyper-parameters of the federated setting were analysed in detail and conclusions were drawn. The distributions shift problem, in particular, was then analysed in detail under two different lights: first related to the change of class distribution and then related to the change in the domains among clients. The first class of problems is formally known as Statistical Heterogeneity and the second as Domain Generalisation. 

Finally, we propose the first, to the best of our knowledge, adaptation of the well-known adversarial learning technique DANN to the Federated scenario as a possible domain generalisation solution.


--------------------------------------------------------------------------------

References: 

[1] Google AI Blog, Federated Learning: Collaborative
Machine Learning without Centralized Training Data.
[2] Gregory Cohen, Saeed Afshar, Jonathan Tapson,
and Andr ́e van Schaik. ”EMNIST: an extension
of MNIST to handwritten letters”. arXiv preprint
arXiv:1702.05373, 2017.
[3] McMahan, Brendan et al. “Communication-Efficient
Learning of Deep Networks from Decentralized
Data.” Proceedings of the 20th International Confer-
ence on Artificial Intelligence and Statistics, PMLR
54: 1273-1282, (2017).
[4] Li, Tian, et al. “Federated Learning: Challenges,
Methods, and Future Directions.” IEEE Signal Pro-
cessing Magazine 37.3 (2020): 50-60.
[5] Kairouz, Peter, et al. “Advances and Open Prob-
lems in Federated Learning.” arXiv preprint arXiv:
1912.04977 (2019).
[6] Tzu-Ming Harry Hsu, Hang Qi, and Matthew Brown.
“Measuring the effects of non-identical data distribu-
tion for federated visual classification.”, 2019.
[7] Hsu TM.H. et al. “Federated Visual Classification with
Real-World Data Distribution. European Conference
on Computer Vision . ECCV 2020. Lecture Notes in
Computer Science, vol 12355. Springer, Cham.
[8] Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan,
Pascal Germain, Hugo Larochelle, Franc ̧ois
Laviolette, Mario Marchand, Victor Lempitsky,
”Domain-Adversarial Training of Neural Networks”.
arXiv:1505.07818 [stat.ML].
[9] Isabela Albuquerque, et al.: ”Adversarial target-
invariant representation learning for domain general-
ization”. arXiv preprint arXiv:1911.00804 (2020)
[10] Caldarola, D., Caputo, B., & Ciccone, M. (2022, Oc-
tober). “Improving generalization in federated learn-
ing by seeking flat minima.” In Computer Vi-
sion–ECCV 2022: 17th European Conference, Tel
Aviv, Israel, October 23–27, 2022, Proceedings, Part
XXIII (pp. 654-672). Cham: Springer Nature Switzer-
land.
[11] Zhou, K., Liu, Z., Qiao, Y., Xiang, T., & Loy, C. C.
(2021). “Domain generalization in vision: A survey.”
IEEE Transactions on Pattern Analysis and Machine
Intelligence (TPAMI), 2022.
[12] Liu, Quande, et al. ”FedDG: Federated domain gener-
alization on medical image segmentation via episodic
learning in continuous frequency space.” Proceedings
of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition. 2021.
[13] Zhang, Liling, et al. ”Federated learning with domain
generalization.” arXiv preprint arXiv:2111.10487
(2021).
[14] Nguyen, A. Tuan, Philip Torr, and Ser-Nam Lim.
”FedSR: A Simple and Effective Domain General-
ization Method for Federated Learning.” Advances in
Neural Information Processing Systems. 2022.
[15] Caldas, Sebastian, et al. ”Leaf: A benchmark for fed-
erated settings.” Workshop on Federated Learning for
Data Privacy and Confidentiality (2019).
[16] Cho, Yae Jee, Jianyu Wang, and Gauri Joshi. ”Client
Selection in Federated Learning: Convergence Analy-
sis and Power-of-Choice Selection Strategies.”, arXiv
preprint arXiv:2010.01243 (2020)
