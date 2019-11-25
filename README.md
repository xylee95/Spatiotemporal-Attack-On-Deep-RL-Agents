# Spatiotemporal-Attack-On-Deep-RL-Agents
---
## Brief overview
This post demonstrates some of the action space attack strategies on Deep RL agents presented in the the paper Spatiotemporally Constrained Action Space Attack on Deep RL Agents. 

The first attack strategy developed was the Myopic Action Space Attack (MAS). In MAS, we formulated the attack model as an optimization problem with decoupled constraints. The objective of the optmization is to craft a perturbation at every step that minimizes the Deep RL agent's immediate reward, subject to the constraint that the perturbation cannot exceed a certain budget. The constraints are decoupled since each constraint is only imposed upon the the immediate perturbation and are independent of the agent's trajectory. Hence, this can be thought of a myopic attack since we only aim to reduce immediate reward without taking into account future rewards. Given a small budget, these attacks might not have any apparent effect on the Deep RL agent as the agent might be able to recover in future steps.

The second attack strategy proposed was the Look-ahead Action Space Attack (LAS). In LAS, we formulated the attack model as an optimization problem with constraints that are coupled through the temporal dimension. The objective of the optimization is to craft a sequence of perturbation that minimizes the Deep RL agent's cumulative reward of a trajectory, subjected to the constraint that the total perturbations in the sequence cannot exceed a budget. By considering an overall budget of perturbations over a trajectory, the crafted perturbations are **less** myopic since they take into account the future states of the agent. Hence, a given budget of perturbations can be allocated more effectively to vulnerable states rather than being forced to expend all the perturbation on the immediate state.

## Results

As hypothesized, given the same budget, LAS proves to be a much stronger attack than MAS, which is in turn stronger than a random perturbation.
![Distribution of rewards for PPO in Lunar Lander agent under different attacks](/images/PPO_LL_boxplot.png "Distribution of rewards for PPO in Lunar Lander agent under different attacks")

## Box2D environments
Trained PPO agent in Lunar Lander Environment
![PPO Agent in Lunar Lander Environment](/images/PPO_LL_nom.gif "PPO agent in Lunar Lander Environment")

Trained DDQN agent in Lunar Lander Environment
![DDQN Agent in Lunar Lander Environment](/images/DDQN_LL_nom.gif "DDQN agent in Lunar Lander Environment")

Implementation of LAS attacks on PPO agent in Lunar Lander Environment
![LAS attack on PPO](/images/PPO_LL_LAS_b4h5.gif "LAS attack on PPO")

Implementation of LAS attacks on DDQN agent trained in Lunar Lander Environment
![LAS attack on DDQN](/images/DDQN_LL_LAS_b5h5.gif "LAS attack on DDQN")

Trained PPO agent in Bipedal Walker Environment
![PPO Agent in Bipedal Walker Environment](/images/PPO_BW_nom.gif "PPO agent in Lunar Lander Environment")

Trained DDQN agent in Bipedal Walker Environment
![DDQN Agent in Bipedal Walker Environment](/images/DDQN_BW_nom.gif "PPO agent in Lunar Lander Environment")

Implementation of LAS attacks on PPO agent trained in Bipedal Walker Environment
![LAS attack on PPO Agent in Bipedal Walker Environment](/images/PPO_BW_LAS_b5h5.gif "PPO agent in Lunar Lander Environment")

Implementation of LAS attacks on DDQN agent trained in Bipedal Walker Environment
![LAS attack on DDQN Agent in Bipedal-Walker Environment](/images/DDQN_PPO_LAS_b5h5.gif "PPO agent in Lunar Lander Environment")

## MUJOCO Environments
Trained PPO agent in Walker-2D

Trained PPO agent in Half-Cheetah

Trained PPO agent in Hopper

PPO agent under LAS attack in Walker-2D

PPO agent under LAS attack in Half-Cheetah

PPO agent under LAS attack in Hopper

More detailed information and supplemental materials are available at https://arxiv.org/abs/1909.02583



## Implementation
### Pre-requisites 
This repository crafts the action space attacks on RL agents. The nominal agents were trained using ChainerRL library. Strictly speaking, the attacks does not require any specific libraries but the code in this repository utilizes Chainer variables and Cupy to accelerate the attacks. 

### Code Structure
1. (agent)_adversary.py contains class of agents that have been augmented to explicitly return Q-values, value functions and action probability distributions.
2. norms.py contains implementations of different norms and projections.
3. (agent)_inference.py contains implementations of attack algorithms on RL agent during inference.

### Running the code
1. To run inference using trained RL agent:
    * Run any one of the (agent)_inference.py with environment arguments 
         * --env_id **LL** for LunarLanderContinuous-v2
         * --env_id **BW** for BipedalWalker-v2
         * --env_id **Hopper** for Hopper-v2 
         * --env_id **Walker** for Walker2d-v2 
         * --env_id **HalfCheetah** for HalfCheetah-v2 
    * Attack argument
         * --rollout **Nominal** runs nominal agent for visualization
         * --rollout **MAS** runs a nominal agent and attacks the agent's action space at every step
         * --rollout **LAS** runs a nominal agent and attacks the agent's action space using attacks that are optimized and projected back to the spatial and temporal budget constraints.
    * Budget of attack
         * --budget any integer or float value
    * Type of spatial projection
         * --s **l1** for l1 projection of attacks onto action dimensions
         * --s **l2** for l2 projection of attacks onto action dimensions
    * Planning horizon (only for LAS)
         * --horizon any integer value
    * Type of temporal projection (only for LAS)
         * --t **l1** for l1 projection of attacks onto temporal dimensions
         * --t **l2** for l2 projection of attacks onto temporal dimensions

Example: To run LAS attack on a PPO agent in Lunar Lander environment with an allocated budget of 5 with a planning horizon of 5 steps using l2 temporal and spatial projections

`python ppo_inference.py --env_id LL --rollout LAS --budget 5 --horizon 5 --s l2 --t l2`

For a list of required packages, please refer to requirements.txt. 

###
# h1 Heading 8-)
## h2 Heading
### h3 Heading
#### h4 Heading
##### h5 Heading
###### h6 Heading
## Horizontal Rules

___

---

***


## Typographic replacements

Enable typographer option to see result.

(c) (C) (r) (R) (tm) (TM) (p) (P) +-

test.. test... test..... test?..... test!....

!!!!!! ???? ,,  -- ---

"Smartypants, double quotes" and 'single quotes'


## Emphasis

**This is bold text**

__This is bold text__

*This is italic text*

_This is italic text_

~~Strikethrough~~


## Blockquotes


> Blockquotes can also be nested...
>> ...by using additional greater-than signs right next to each other...
> > > ...or with spaces between arrows.


## Lists

Unordered

+ Create a list by starting a line with `+`, `-`, or `*`
+ Sub-lists are made by indenting 2 spaces:
  - Marker character change forces new list start:
    * Ac tristique libero volutpat at
    + Facilisis in pretium nisl aliquet
    - Nulla volutpat aliquam velit
+ Very easy!

Ordered

1. Lorem ipsum dolor sit amet
2. Consectetur adipiscing elit
3. Integer molestie lorem at massa


1. You can use sequential numbers...
1. ...or keep all the numbers as `1.`

Start numbering with offset:

57. foo
1. bar


## Code

Inline `code`

Indented code

    // Some comments
    line 1 of code
    line 2 of code
    line 3 of code


Block code "fences"

```
Sample text here...
```

Syntax highlighting

``` js
var foo = function (bar) {
  return bar++;
};

console.log(foo(5));
```

## Tables

| Option | Description |
| ------ | ----------- |
| data   | path to data files to supply the data that will be passed into templates. |
| engine | engine to be used for processing templates. Handlebars is the default. |
| ext    | extension to be used for dest files. |

Right aligned columns

| Option | Description |
| ------:| -----------:|
| data   | path to data files to supply the data that will be passed into templates. |
| engine | engine to be used for processing templates. Handlebars is the default. |
| ext    | extension to be used for dest files. |


## Links

[link text](http://dev.nodeca.com)

[link with title](http://nodeca.github.io/pica/demo/ "title text!")

Autoconverted link https://github.com/nodeca/pica (enable linkify to see)


## Images

![Minion](https://octodex.github.com/images/minion.png)
![Stormtroopocat](https://octodex.github.com/images/stormtroopocat.jpg "The Stormtroopocat")

Like links, Images also have a footnote style syntax

![Alt text][id]

With a reference later in the document defining the URL location:

[id]: https://octodex.github.com/images/dojocat.jpg  "The Dojocat"


## Plugins

The killer feature of `markdown-it` is very effective support of
[syntax plugins](https://www.npmjs.org/browse/keyword/markdown-it-plugin).


### [Emojies](https://github.com/markdown-it/markdown-it-emoji)

> Classic markup: :wink: :crush: :cry: :tear: :laughing: :yum:
>
> Shortcuts (emoticons): :-) :-( 8-) ;)

see [how to change output](https://github.com/markdown-it/markdown-it-emoji#change-output) with twemoji.


### [Subscript](https://github.com/markdown-it/markdown-it-sub) / [Superscript](https://github.com/markdown-it/markdown-it-sup)

- 19^th^
- H~2~O


### [\<ins>](https://github.com/markdown-it/markdown-it-ins)

++Inserted text++


### [\<mark>](https://github.com/markdown-it/markdown-it-mark)

==Marked text==


### [Footnotes](https://github.com/markdown-it/markdown-it-footnote)

Footnote 1 link[^first].

Footnote 2 link[^second].

Inline footnote^[Text of inline footnote] definition.

Duplicated footnote reference[^second].

[^first]: Footnote **can have markup**

    and multiple paragraphs.

[^second]: Footnote text.


### [Definition lists](https://github.com/markdown-it/markdown-it-deflist)

Term 1

:   Definition 1
with lazy continuation.

Term 2 with *inline markup*

:   Definition 2

        { some code, part of Definition 2 }

    Third paragraph of definition 2.

_Compact style:_

Term 1
  ~ Definition 1

Term 2
  ~ Definition 2a
  ~ Definition 2b


### [Abbreviations](https://github.com/markdown-it/markdown-it-abbr)

This is HTML abbreviation example.

It converts "HTML", but keep intact partial entries like "xxxHTMLyyy" and so on.

*[HTML]: Hyper Text Markup Language

### [Custom containers](https://github.com/markdown-it/markdown-it-container)

::: warning
*here be dragons*
:::


