In this package are developped different loss functions:

## Lagrangian Sub-Problem Loss

`loss_LR` is the one presented in the paper [^1] and consist on the bound provided by the Lagrangian Sub-Problem (with a proper sign that allows to write the Lagrangian Dual as minimization problem).


## Lagrangian Sub-Problem Loss on GPU

`loss_LR_gpu`: as `LRloss` but the sub-problem is solved in GPU.
### Warning: for the moment this loss function works only for MCND instances with GPU memorization. 
Anyway it is not faster than the one that use CPU.

## GAP Loss
`loss_GAP` this loss function simply consists on the GAP of percentage  the value ``v`` provided by the Lagrangian sub-Problem and the optimal value of the Lagrangian dual and can be computed as:
```math
\frac{v-v^*}{v^*}*100
```
when the Lagrangian Dual is a minimization problem

## GAP Closure Loss
 `loss_GAP_closure` this loss function is similar to `GAPloss` as still consider the value ``v`` provided by the Lagrangian sub-Problem and the optimal value of the Lagrangian dual. But it tries to further compare these solutions with the continuous relaxation bound `CR`.
It is computed as:
```math
\frac{v}{v^*-CR} * 100
```
## Hinge Loss

For an instance ``\iota \in I`` with gold solution ``(x^*, y^*)`` (more precisely ``(x^*(\iota), y^*(\iota))``) of ``L(\pi^*)``, the Hinge loss is
```math
H(w;\iota) =  L(\pi(w), x^{*}, y^{*}) - \min_{x,y} \Big(L(\pi(w), x, y) - \alpha \Delta_{y^*}(y)\Big)
```
where ``w`` are the parameters of the model, ``\pi(w)`` is the prediction of the model given ``w``, ``(x^{*},y^{*})``  is the gold solution of the instance, 
``\Delta_{y^*}(y)`` is the hamming loss between ``y`` and ``y^*`` and ``\alpha`` is a non negative scalar.

The `loss_hinge` is 
```math
\frac{1}{|I|}\sum_{\iota \in I} \frac{1}{|A(\iota)|}H(w;\iota)
```

## Mean Squared Error

`loss_mse` is only the MSE between the mean squared error between the predicted and the optimal Lagrangian Multipliers.

## Multi Prediction LR loss

`loss_multi_LR_factory` is a specialized version of `loss_LR` that is able to handle with Multiple Lagrangian Multipliers Predictions.

### More sophisticated variants of this loss function will be provided in the future. 

## References

[^1]: F. Demelas, J. Le Roux, M. Lacroix, A. Parmentier "Predicting Lagrangian Multipliers for Mixed Integer Linear Programs", ICML 2024.























