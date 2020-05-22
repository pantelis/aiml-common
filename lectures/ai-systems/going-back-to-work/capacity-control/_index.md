---
title: Capacity Control
weight: 142
draft: true
---

# Capacity Control

## Assumptions

To simplify the problem, 

1. The departure times cannot change dynamically. This means that if the actual demand is lower than the predicted demand when the scheduling decision took place, the departure time stays fixed. Employees / customers can extend their departure by sending another request via the messaging app but this is an independent reservation and is not unaccounted for the decision policy. 
   
2. Although this can be very useful for stores and co-working spaces, there is no attempt to flatten the demand curve by distributing accepts among geographically separated facilities of the same legal entity (e.g. Home Depot stores). We will extend treatment to network capacity control that implements load balancing. 

3. There is no semantic consideration of the enterprise business type in the decision making algorithm. For example, we do not consider special circumstances such as theaters and other sports venues as they control the reservations and seating on their own to maintain social distancing. 
   
4. There is no dynamic tracking of traffic within the building. The system is not designed (although it can be extended via video surveillance and other worker localization technologies) to consider worker mobility and congestion events from people dont adhering to social distancing guidelines. 

5. There are no fairness considerations. One can go ahead and book multiple time slots by sending messages one after the other to the system. This is an important limitation but can be easily addressed in practice and will unnecessarily complicate the policy.

6. There are no risk considerations in the decision making process. A user that requests a reservation that lives far away and travels via public transportation (higher risk of infecting others) has equal chances to a user that lives next to the requested address (lower risk). 

## Terminology and Notation

| Term   | Definition  |
| --- | --- |
|  Host   | A host can be an enterprise, building owner, space owner, school etc. A host is the account owner of the SaaS app  and via access rights she can adjust the total available capacity and other parameters that affect the policy. |
| Site | A host is responsible for the management of multiple sites. A site can be a building. |
| Resource | A site has multiple resources that are managed by the application. A resource can be a floor or a room where multiple workers need to use at the same time. A room can also be an open floor plan.  |
|  Worker   | The worker is a person that requests admission in the physical space of the host.  |
| Group | A group os a number of workers that request joint admission. | 
|  $C$   |  Maximum available capacity / occupancy per site resource.  $C$ is a single knob at the disposal of either the host or is controlled by federal or local governments based on the safe occupancy limits of each site / resource. These limits are well documented in the license that many hosts need to obtain for them to legally operate.  $C$obviously varies from host to host. |
|     |     |
|     |     |
|     |     |
|     |     |
|     |     |
|     |     |

## Revenue Management for Capacity Control

In this section we outline a capacity control policy that is routinely used in various industries (airlines, car rentals, hospitality) to make reservations towards perishable capacity-constrained resources. 

Each host is assumed that is organized according to an org chart - a tree structure that lists  _actors_ and in the problem we are addressing here we include the customers/students as workers. For an organization to operate successfully certain workers in this tree must enjoy guaranteed admission. In a office setting for example, certain workers whose function is to preserve security are essential. Note that each host can impose via the SaaS app, admission dependencies between workers. Some businesses must be able to admit an worker only if there are admission for others. For example, there is no point to admit far more customers into a store if the necessary cashiers are not in place. Similarly at a university setting, there is no point of admitting student workers if faculty and TA workers needed for a scheduled class are not admitted. 

For the reasons above, we establish the notion of _actor class_ such as manager, employee, student. The _minimum_ number of classes we can have is two eg. faculty class and student class. 


### Two Class Model

Let us assume that we are interested in allocating a resource of capacity $C$, to two different types of workers.Each type is assigned a price that workers will need to pay for one unit of this resource. Note that this payment is fictitious - it just represents the _value_ of each worker type to the organization. Let $p_d$/$D_d$ and $p_f$/$D_f$ represent the prices and demands for the discounted / full-price types respectively. The assumptions behind our model are:


* $p_d < p_f$
* $D_d$ and $D_f$ are independent random variables i.e. $p(D_d|D_f) = p(D_d)$.
* The type d demand arrives before type-f demand. This time-element is not important at this point.

Our aim is to find the a-priori protection level $y$ that represents the resource units that will be reserved for allocation to the type $f$ traffic. 

The maximum number of resource units that can be allocated to type $f$, i.e. the **available capacity** for type $d$, will then be,

$$C_d = C-y$$

The units that can be occupied by type $d$ cannot exceed the corresponding demand for this type i.e.,

$$S_d = \min (C_d,D_d)$$

Therefore the maximum number of resource units that can be allocated to type $f$, i.e. the {\bf available capacity} for type $f$, will then be,

$$C_f = C - \min (C_d,D_d) = \max(y,C-D_d)$$ 

The units that will be occupied by type $f$ is then,

$$S_f = \min (C_f,D_f)$$

The expected fictitious revenue from the occupied resources will be 

$$\mathbb E \\{ p_d  S_d + p_f  S_f \\}$$ 

where the expectation is taken over the random demands of the two types. It represents the _return_ to the establishment from the physical presence of all admitted workers. If we maximize revenue over the protection level $y$, we obtain 

$$
V_d(C) = \max_{0 \le y \le C} \mathbb E \\{p_d  S_d + p_f  S_f \\}  = \max_{0 \le y \le C} \mathbb E \\{p_d  \min (C-y,D_d) + p_f \min \left( \max(y,C-D_d),D_f \right) \\} 
$$

Intuitively, the ratio of the prices $r=\frac{p_d}{p_f}$ can help us determine some trends in setting the reservation level $y$. For if the ratio is very small, i.e. $p_f >> p_d$, then we would be inclined to reserve most of the overall capacity $C$ for type $f$ i.e $C_f >> C_d$. If on the other hand the ratio is close to 1, then we would be inclined to reserve only a very small capacity for type $f$ i.e. $C_f << C_d$, since we can get almost the same revenue with type $d$. 

Let us assume now that we set a fixed price ratio $r$ that is not close to these two extremes, and we then look at the remaining factors that can determine $y$ for optimal revenue. From the expressions above that factor seems to be the shape of the tail of the demand function for type $f$ i.e. $p(D_f > y)$. This is because we can only have the revenue for the capacity we reserved, if the associated demand is there i.e. if $D_f > y$. In the next section we describe one technique based on Dynamic Programming (DP) that can be used to calculate the optimal reservation/protection limit. 


## Dynamic Programming Solution

To use a dynamic programming problem formulation, we need to define the variables at the beginning of the two periods involved. The first period is defined as the point just before the type $d$  demand is observed i.e. when the available capacity is C and we define the value function $V_d(C)$ that represents the optimal expected revenue starting with $C$ units of capacity. Apparently an upper bound of this value function is $V_d(C) \le p_f C$, and this is obtained when we protect all the available capacity for type $f$.

Similarly, the second variable is the value function $V_f(x)$ that represents the optimal expected revenue starting with $x$ units of capacity just before observing $D_f$ or equivalently after we observe $D_d$. 

$$V_f(x) = \mathbb{E} \\{p_f  \min(x,D_f)\\}  =  p_f  \sum_{j=1}^x p(D_f \ge j)$$

where we have expanded $\mathbb E \\{\min(x,D) \\}$ using the expression derived in the appendix. 

The dynamic program formulation would involve relating $V_d(C)$ to $V_f(x)$. If we rewrite $V_d(C)$ from the result of the previous section,

$$ V_d(C)  =  \max_{0 \le y \le C}  \mathbb E \\{p_d \min (C-y,D_d)\\} +  \mathbb E \\{ V_f(C_f) \\}  = $$

$$ =  \max_{0 \le y \le C}    \mathbb E \\{ p_d \min (C-y,D_d) \\} +  \mathbb E \\{ V_f(\max(y,C-D_d)) \\} = \max_{0 \le y \le C} W(y,C)$$

The revenue difference from one unit of protection level is given by,

$$\Delta W(y,C) = W(y,C) - W(y-1,C) = [p_f p(D_f \ge y) - p_d] p(D_d \ge C-y)$$

{{<details "Click here for the proof of $\Delta W(y,C)$">}}

We can prove the last equality as follows

$$ W(y,C) = \mathbb E \\{p_d \min (C-y,D_d) + V_f(\max(y,C-D_d)) \\} $$

$$W(y-1,C) = \mathbb E \\{p_d \min (C-y+1,D_d) + V_f(\max(y-1,C-D_d)) \\}$$

We can study two cases: 

| Case    | Description   |
| --- | :-- |
| $D_d \le C-y$ |  $ \Delta W(y,C) = W(y,C) - W(y-1,C)$ | 
| | $ = \mathbb E \\{p_d D_d + V_f(C-D_d) \\} - \mathbb E \\{p_d D_d + V_f(C-D_d)\\} $ |
| | $ = \mathbb E \\{0 p(D_d \le C-y) \\} =0 $ | 
| $D_d > C-y$ | $ \Delta W(y,C) = W(y,C) - W(y-1,C)$ | 
| | $= \mathbb E \\{p_d (C-y) + V_f(y) \\} - \mathbb E \\{p_d (C-y+1) + V_f(y-1) \\}$ |
| | $= \mathbb E \\{ V_f(y)-V_f(y-1)-p_d \\} p(D_d > C-y)$ |

Then in all cases, the difference,


$\Delta W(y,C) = W(y,C) - W(y-1,C)$
$= \mathbb E \\{V_f(y)-V_f(y-1)-p_d \\} p(D_d > C-y)$
$= \mathbb E \\{\Delta V_f(y)-p_d \\} p(D_d > C-y)$

If we replace the marginal of the value function $V_f(y)$, with $\Delta V_f(y) = V_f(y)-V_f(y-1) = p_f  p(D_f \ge y)$ we get the final result:

$$\Delta W(y,C) = W(y,C) - W(y-1,C) = \mathbb E \\{ p_f  p(D_f \ge y)-p_d \\} p(D_d > C-y)$$

The term is $\Delta V_f(y)- p_d$ will start positive and then become negative i.e. the marginal value will have at least a local maximum. This can be simply shown if one replaces $y$ with $\infty$ which causes the term $p_f  p(D_f \ge y)-p_d \rightarrow -p_d$ and in the other extreme, if we replace $y$ with $0$ the term $p_f  p(D_f \ge y)-p_d \rightarrow p_f-p_d$. 

Note that the marginal value $V_f(x)$ itself reduces with remaining capacity $x$. The marginal value depends on how much the tail of the type-$d$ demand exceeds the available capacity. 

{{</details>}}


### Calculating the Optimal Protection Limit and Maximum Revenue

The optimal $y$, denoted by $y^*$ can now be found as,

$$y^* = \max \\{ y \in N_+: p_f  p(D_f \ge y^*) \ge p_d \\} = \max \\{ y \in N: p(D_f \ge y^*) \ge \frac{p_d}{p_f} \\}$$

The optimal booking limit will then be 

$$b^* = (C-y^*)^+$$

The maximum possible revenue starting with capacity $C$ is given by,

$$V_d(C) =  \begin{cases} W(y^*,C) & \text{if $y^* \le C$,} \\\\ W(C,C) &\text{if $y^* > C$.} \end{cases}$$

Calculating $W(y^*,C)$ can be done recursively. We know that, 

$$W(y,C)  =  W(y-1,C)  + \mathbb E \\{p_f  p(D_f \ge y)-p_d \\} p(D_d > C-y)$$

To start the iteration we need $W(0,C)$ that can be written as,

$$W(0,C) = p_d  \mathbb E \\{\min(C,D_d)\\} + \mathbb E \\{V_f(\max(0,C-D_d))\\}$$

The first expectation can be written using partial expectations as,

$$\mathbb E \\{\min(C,D_d)\\} = \sum_{j=1}^C p(D_d \ge j)$$

The second expectation can be written as,

$$\mathbb E \\{V_f(\max(0,C-D_d))\\} = \sum_{j=0}^C V_f(C-j) p(D_d = j)$$

Then, starting from $W(0,C)$ we can calculate $W(1,C)$ and iterate until all $W(y^\*,C)$ are calculated. 

### Calculation of the Spill Rate

The spill rate is defined as the portion of type $f$ traffic that is rejected given the revenue maximizing optimal protection level $y^\*$. Class $f$ is rejected when the corresponding demand exceeds the available capacity. From the previous sections we know that the available capacity if $C_f = C - \min (C_d,D_d) = \max(y^*,C-D_d)$. Then, type $f$ will be rejected with probability (the spill probability) equal to,

$$p(D_f > \max(y^*,C-D_d))$$

An upper bound of the spill probability can be easily obtained if we consider the definition of $y^\*$ as the largest $y$ that $p(D_f \ge y^\*) \ge \frac{p_d}{p_f}$. Given that $\max(y^\*,C-D_d) > y^\*$, we can also  write the relationship between the two tail probabilities as, $p(D_f \ge y^\*) \ge p(D_f > \max(y^\*,C-D_d))$. 

Therefore the upper bound is

$$p(D_f \ge y^\*) < \frac{p_d}{p_f}$$

When the demand for the type $d$ is large, then the spill rate will get very close to the ratio $\frac{p_d}{p_f}$.


### Comparative Statics

If we assume that the demand distribution $D_f$ is Normal i.e. $N(\mu_f,\sigma_f)$, from the previous section we know that the optimal protection limit can be found as the solution of the equation,


$$p(D_f \ge y^*) = \frac{p_d}{p_f} 1-\Phi(\frac{y^*-\mu_f}{\sigma_f}) = \frac{p_d}{p_f}$$

and solving for $y^\*$ we obtain,

$$y^* = \mu_f + \sigma_f \Phi^{-1}(1-\frac{p_d}{p_f})$$

Such protection limit makes intuitive sense in that for a demand function that is perfectly known i.e. it is deterministic, the $\sigma_f=0$ and then the optimal $y^* = \mu_f$. For the more general case it will be larger than $\mu_f$ when the $\Phi^{-1}(1-\frac{p_d}{p_f}) >0$ i.e. when $1-\frac{p_d}{p_f} > \frac{1}{2}$. This is the case when $p_d < p_f/2$.It is straightforward to conclude that if $p_d = p_f/2$ then $y^* = \mu_f$. Similarly, if $p_d > p_f/2$ then $y^* < \mu_f$. 

It is also evident that when $\sigma_f$ is large, the protection level deviates further from the mean. So when we observe protection levels that are around the mean, we can conclude that either the variance is small or the price ratio is close to $\frac{1}{2}$.

### Dependent Demands

Demand variables may be dependent due to a variety of factors. Suppose that traffic of type $d$ after rejection can decide to pay the price of type $f$ traffic and place itself in the queue for type $f$ resources. The rejected type $d$ traffic is given by,

\begin{equation}
R_d = (D_d,(C-y^*))^+
\end{equation}

If $\theta$ is the probability that the rejected type $d$ traffic is willing to pay price $p_f$, the residual demand that originate from the rejected traffic is,

\begin{equation}
U(\theta,y) = Bin(\theta,R_d)
\end{equation}

where Bin is the binomial distribution. This means that the updated type $f$ demand is given by,

\begin{equation}
	D_f(y)=D_f + U(\theta,y)
\end{equation}

Later we will show that if $\theta > \frac{p_d}{p_f}$, we will protect all resources for type $f$. 


\section{The Multiple Class Model}
The multiple type model is explained in the corresponding notes by Prof. Gallego. In this section we will focus on the computational aspects and the associated Matlab model developed. 

The key DP equation is the calculation of the value function. This is given by,

\begin{eqnarray}
V_j(x) &=& \sum_{k=0}^{x-y-1} \left[p_j k + V_{j-1}(x-k) \right] p(D_j=k) \\
&+& \left[p_j(x-y) + V_{j-1}(y)\right] p(D_j \ge x-y)
\end{eqnarray}

This equation was implemented in Matlab as shown in the next listing.

{{<details "Matlab Code for Value Function" >}}
```matlab
function Resource = value_function(Resource,stage,inpstruct)

% vector of all potential states (states represent the 
% remaining capacity)
x=1:inpstruct.Num_States;


% the pdf of the demand for this stage - P(D_j == x)
demand_pdf = poisspdf(x,inpstruct.mean_D(stage));

% the complementary cdf of the demand for this stage - P(D_j >= x)
demand_ccdf = 1-poisscdf(x,inpstruct.mean_D(stage));

if stage==1
    % The optimal protection level for this initial stage
    y_star_prev = 0;
else
    % The optimal protection level for this and subsequent stages
    y_star_prev = Resource{stage-1}.Protection_Level;
end
T=zeros(inpstruct.C,inpstruct.C);
% Calculate the value function V (note that we are reusing x defining 
%it as the state index)
for x=1:inpstruct.Num_States
    Resource{stage}.T(x,:)=0;
    for k=1:x-y_star_prev        
        T(x,k)= (inpstruct.price(stage) *  k + ...
        Resource{stage-1}.V(max(1,x-k)))* demand_pdf(k);
    end
    Resource{stage}.V(x) = sum(T(x,:)) + ...
            (inpstruct.price(stage)* max(0,x-y_star_prev) + ...
             Resource{stage-1}.V(max(1,min(x,y_star_prev)))) *...
                demand_ccdf(max(1,x-y_star_prev));      
end

% Calculate the marginal value of a single resource unit (DeltaV)
deltaV = diff(cat(2,0,Resource{stage}.V));
Resource{stage}.deltaV= deltaV;
% find the point of where the marginal value is larger than the price of
% the next stage 
indeces = find(deltaV > inpstruct.price(stage+1));
Resource{stage}.Protection_Level = max(indeces);
```
{{</details>}}

A number of calls equal to the number of stages are made. In each call, the corresponding results are stored for usage at the next stage. Using the software in the problem defined in Table \ref{tbl:Lecture_notes_week2_example1} that is the Example 1 of the multiple type lecture notes, 


\begin{table*}[htbp]
	\centering
		\begin{tabular}{cccc}
		\hline
			{\textbf Class} & \textbf{Price} & \textbf{Demand Distribution} & \textbf{Optimal Protection Levels} \\ \hline
			1 & 100 & Poisson(10) & 6 \\
			2 & 90 & Poisson(15) & 20 \\
			3 & 80 & Poisson(25) & 44 \\
			4 & 70 & Poisson(15) & 50 \\			\hline
		\end{tabular}
	\caption{Example 1 of Lecture Notes: Available initial capacity $C=50$ and 4 demand types.}
	\label{tbl:Lecture_notes_week2_example1}
\end{table*}


Fig.\ref{fig:marginal_value_1} shows the marginal value as a function of the remaining resources for the three stages of the problem.

\begin{figure}[htb]
\centering{
\includegraphics[height=3in,angle=0]{Marginal_Value_DP-StaticLecture_notes_week2_example1.pdf}}
\caption{Decreasing Marginal Value as the Remaining Resources Increase for the Table \ref{tbl:Lecture_notes_week2_example1}  Problem}
\label{fig:marginal_value_1}
\end{figure}

The optimal protection limits are shown in Fig. \ref{fig:protection_limits_1}. 

\begin{figure}[htb]
\centering{
\includegraphics[height=3in,angle=0]{Optimal_Protection_Limits_DP-StaticLecture_notes_week2_example1.pdf}}
\caption{Optimal Protection Limits for the Table \ref{tbl:Lecture_notes_week2_example1} Problem}
\label{fig:protection_limits_1}
\end{figure}

In the example of the provided Excel spreadsheet for two types with Poisson arrivals as in Table \ref{tbl:two_class}, 


\begin{table*}[htbp]
	\centering
		\begin{tabular}{cccc}
		\hline
			{\textbf Class} & \textbf{Price} & \textbf{Demand Distribution} & \textbf{Optimal Protection Levels} \\ \hline
			1 & 1000 & Poisson(40) & 41 \\
			2 & 450 & Poisson(15) & 100 \\		
		\end{tabular}
	\caption{Problem parameters: Available initial capacity $C=100$ and 2 demand types.}
	\label{tbl:two_class}
\end{table*}

Fig.\ref{fig:marginal_value_2} shows the marginal value as a function of the remaining resources.

\begin{figure}[htb]
\centering{
\includegraphics[height=3in,angle=0]{Marginal_Value_DP-StaticTwo-Class.pdf}}
\caption{Decreasing Marginal Value as the Remaining Resources Increase for the Table \ref{tbl:two_class} Problem}
\label{fig:marginal_value_2}
\end{figure}

The optimal protection limits are shown in Figure \ref{fig:protection_limits_2}.

\begin{figure}[htb]
\centering{
\includegraphics[height=3in,angle=0]{Optimal_Protection_Limits_DP-StaticTwo-Class.pdf}}
\caption{Optimal Protection Limits for the Table \ref{tbl:two_class} Problem}
\label{fig:protection_limits_2}
\end{figure}

Its noteworthy to highlight the decreasing marginal value of capacity as the remaining capacity increases. This is intuitive as the more capacity we have remaining the less a unit of this capacity should be worth. Also, it is interesting to highlight the increasing value of marginal capacity with increasing stage. This is also intuitive as the earlier we are in stage i.e. the more time we have until we reach the final stage 1, the more a unit of capacity is worth if we allocate it at that stage.

\section{Bound and Approximation to $V_j(x)$}
In this section we provide an upper bound and a close approximation to the value function $V_j(x)$ that serves in cases where solving the DP will be computationally difficult. We introduce some new notation based on the quantities we have been dealing with in the previous section.

\begin{itemize}
\item $n$ is the number of types
\item $\mu_k = E\{D_k\}, k=1, \dots ,n$
\item $D[1,j]=\sum_{k=1}^{j} D_k$ is distributed as $N(\nu_j,\tau_j)$
\item $\nu_j = E\{D[1,j]\}$
\item $\tau_j=Var\{D[1,j]\}$
\item As previously, $p_1 > p_2 > \dots p_n$ and the demands $D_k$ are independent random variables.
\end{itemize}

\subsection{A Deterministic Upper Bound}
In the deterministic case, we know perfectly the demands $D_k$ i.e. $D_k=\mu_k$. An upper bound on the value function i.e. $V_j(x) \le V_j^d(x)$ is obtained by solving the well known Knapsack problem that can be posed for this case as,

\begin{eqnarray*}
V_j^d(x) &=& \max_b \sum_{k=1}^j p_k b_k \\
&s.t.& \sum_{k=1}^j b_k \le x \\
& & 0 \le b_k \le \mu_k, k=1,...,j\\
\end{eqnarray*}

where the decision variables are the booking limits $b_k$. The optimal booking limits at the $j$-th stage refers to the maximum number of resources that can be occupied by all past types including the current one $n,\dots,j$. Recall that the optimal booking limits are related to the optimal protection levels by the expression $b_j^* = x - y_{j-1}^*, j=2,\dots,n$, with $b_1^*=x$. In other words the total capacity $x$ at each stage $j$ can be divided into $b_j^*$ and $y^*_{j-1}$ the protection limit for all future types $j-1,\dots,1$.

If we have ample capacity i.e. $C \ge \nu_n$ then we can allocate all the known arriving demand at each stage greedily. This means that at stage 1 we will allocate a booking limit equal to the demand at this stage i.e. $b_1=\mu_1$ exhausting all type 1 demand that offers the highest revenue increase and proceed by satisfying type-2 demand with the remaining capacity and so on until the demand is exhausted. Then the value function can be written as,

\begin{equation}
V_n^d(C) = \sum_{k=1}^n p_k \mu_k
\end{equation}

In terms of protection levels, by definition and for $j=1 \dots n$,

\begin{equation}
y_{j}^* = x - \mu_{j+1}^* = \sum_{k=1}^{j+1} \mu_k -\mu_{j+1} = \nu_j
\end{equation} 

If on the other hand $C < \nu_n$, the greedy solution to the knapsack problem will be $b_k^* = \mu_k, k=1,\dots,j-1$ and for the last stage $j(x) < n$ where the capacity $x$ is exhausted, it will be $b_j=x-\nu_{j-1}$. The protection levels will be just like in the previous case until the stage where the capacity is not exhausted i.e. $y_{k}^*  = \nu_k$ for $k=1,\dots,j(x)-1$ and for the last stage $k=j(x)$ it will be $y_j^*=x$. Given these relationships the value function can then be written as,

\begin{eqnarray}
V_n^d(x) &=& p_{j(x)}(x-\nu_{j(x)-1}) + \sum_{k=1}^{j(x)-1} p_k \mu_k \\
&=& x p_{j(x)} + \sum_{k=1}^{j(x)-1} (p_k-p_{j(x)}) \mu_k
\end{eqnarray} 

As an example, consider the following KP,

\begin{eqnarray*}
V_3(10) &=& \max_b 100 b_1 + 80 b_2 + 60 b_3 \\
&s.t.& b_1+b_2+b_3 \le 10 \\
& & b_1 \le 5 \\
& & b_2 \le 7 \\
& & b_3 \le 15 \\
\end{eqnarray*}

It is obvious that $j(10)=2$ since the capacity is exhausted in the 2nd stage where the demand is $5+7 > 12$. The optimal protection levels that result from the greedy allocation given the capacity constraint $C=10$, are $b_1^* = 5,b_2^* = 5,b_3^* = 0$. The optimal protection levels are $y_1^*=5, y_2^*=12$. The revenue is $100 \times \mu_1 + 80 \times (10-\mu_1)$ where $\mu_1=5$ for this problem.


\subsection{Approximation to Value Function}
The previous section established an upper bound on the value function i.e. $V_j(x) \le V_j^d(x)$ with the intuitive justification that we can never exceed the revenues of the deterministic problem if we have a finite variance in the problem parameters. In this section we provide an approximation to the value function, denoted as $V_j^a(x)$ that can be calculated without the use of DP. 

We start by defining the average price can be defined as the ratio between the deterministic value function (the max revenue at stage $j$ starting with $x$ units of capacity) and the aggregate demand as observed up to stage j. 

\begin{equation}
p_j^d(x) = \frac{V_j^d(x)}{\min(x,\nu_j)}
\end{equation}

$V_j^d(x)$ can then be written as,

\begin{equation}
V_j^d(x) = p_j^d(x) \min(x,\nu_j)
\end{equation}

Using the Jensen's inequality, 

\begin{equation}
\min(x,E\{Y\}) \ge E\{\min(x,D)\}
\end{equation}

and the fact that $\nu_j = E\{D[1,j]\}$ we can write that,

\begin{equation}
V_j^d(x) = p_j^d(x) \min(x,E\{D[1,j]\}) \ge p_j^d(x) E\{\min(x,D[1,j])\} = V_j^a(x)
\end{equation}

When $x > \nu_j$ then a special case of this approximation can be obtained that uses a different expression for the average price,

\begin{equation}
\bar{p}_{j(x)} = \sum_{k=1}^{j(x)} p_k \frac{\mu_k}{\nu_{j(x)}}
\end{equation} 

The approximation with this average price is,

\begin{equation}
V_j^a(x) =  \bar{p}_{j(x)} E\{\min(x,D[1,j])\}
\end{equation}


The last approximation has the most practical value. Lets now try to calculate the marginal approximate value function with respect to remaining capacity i.e.  $\frac{\partial{V_j^a(x)}}{\partial x}$. There is a need to take the partial derivative of an expectation of a $\min$ expression of $x$. We can think of the expectation as an integral over the density of the random demand. If $x$ becomes greater and already $x > D[1,j]$ then nothing changes and the derivative is 0. If on the other hand $D[1,j] \ge x$, the derivative will be 1.0 since $\min(x,D[1,j])=x$. Therefore the partial derivative will be,

\begin{equation}
\frac{\partial{V_j^a(x)}}{\partial x} =  \bar{p}_{j(x)} p(D[1,j] \ge x)
\end{equation}

The last expression is directly connected to the expression of protection limits obtained by the EMSR-b heuristic, i.e.

\begin{equation}
y_j = \max\{y \in {\cal N}: p(D[1,j] \ge y) > \frac{p_{j+1}}{\bar{p}_j}\} 
\end{equation}

