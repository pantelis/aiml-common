---
title: Backpropagation Exercise Solution
---

# Backpropagation Exercise Solution

Forward path:

(1) $z^{(1)} = W^{(1)} flatten(X^{(1)})$ - if they didnt write "flatten" consider it OK. 

(2) $a^{(1)} = \max(0, z^{(1)})$

(3) $z^{(2)} = W^{(2)} a^{(1)})$

(4) $\hat{y} = \mathtt{softmax}(z^{(2)})$

(5) $L = CE(y, \hat{y})$

Backwards path: 

(5) $\partial L / \partial L = 1.0$

(4) Using the LUT of the gates (last three rows), they can go straight into $\partial L / \partial z^{(2)} = \hat y - y$

(3a) $\partial L / \partial W^{(2)} = a^{(1)} (\hat y - y)$

(3b) $\partial L / \partial a^{(1)} = W^{(2)} (\hat y - y)$

(2) $\partial L / \partial z^{(1)} = \partial L / \partial a^{(1)}$ if   $a^{(1)} > 0$

(1) $\partial L / \partial W^{(1)} = \partial L / \partial z^{(1)} \times flatten(X^{(1)})$