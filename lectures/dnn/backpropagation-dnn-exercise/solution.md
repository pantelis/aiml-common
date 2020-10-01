---
title: Backpropagation Exercise Solution
---

# Backpropagation Exercise Solution



| Forward Pass Step | Symbolic Equation 
| --- | --- |
| (1) | $z^{(1)} = W^{(1)} x^{(1)}$ |
| (2) | $a^{(1)} = \max(0, z^{(1)})$| 
| (3) | $z^{(2)} = W^{(2)} a^{(1)}$| 
| (4) | $\hat{y} = \mathtt{softmax}(z^{(2)})$| 
| (5) | $L = CE(y, \hat{y})$| 


| Backward Pass Step | Symbolic Equation 
| --- | --- |
| (5) | $\frac{\partial L}{\partial L} = 1.0$ | 
| (4) | $\frac{\partial L}{\partial z^{(2)}} = \hat y - y$|
| (3a) | $\frac{\partial L}{\partial W^{(2)}} = a^{(1)} (\hat y - y)$|
| (3b) | $\frac{\partial L}{\partial a^{(1)}} = W^{(2)} (\hat y - y)$|
| (2) | $\frac{\partial L}{\partial z^{(1)}} = \frac{\partial L}{\partial a^{(1)}}$ if   $a^{(1)} > 0$|
| (1) | $\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial z^{(1)}} \times x^{(1)}$|