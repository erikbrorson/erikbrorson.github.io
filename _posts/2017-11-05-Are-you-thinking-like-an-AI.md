---
title: "Are you thinking like an AI?"
layout: post
---


Let's play a little game. Given four words, pick the one that is the least like the other. For example, given the word *Queen*, *King*, *Prince*, and *Cleaner*. Which word would you pick? Personally, I would say that *Cleaner* is the odd one as the other three are related to royalty and the fourth isn't. I think most people – at least those who share some of my cultural biases – would agree with me. Now imagine if we could ask a computer the same question. Would it pick the same word? Let's return to this query later.

Let's stop for a moment and think about what it actually means to participate in the aforementioned game. We take four words as input and output one word which we think is the most unlike the other. We evaluate each word in relation to the others and might take things as spelling, connotations and semantics into consideration. The words are evaluated in the context of the others. We might think of a king and think of royalty, castles, and power. We know that a king is married to a queen and that their son will be called prince. The choice of word in this example is trivial. 

Let's return to the idea of a computer that is able to perform this task. I built small program in Python that achieves this. I will not go into any technical details in this blog post but figured that it might be interesting to list a few of the choices my program made. Feel free to compare, would you make the same decisions as the AI?

|Example  | Words                                     | Pick of the AI      |
|---------|-------------------------------------------|---------------------|
|1.       | Addition, Christmas, Integral, Pythagoras | Christmas           |
|2.       | Am, I, Is, The                            | The                 |
|3.       | Bicycle, Cinnamon, Pump, Wheel            | Cinnamon            |
|4.       | Companionship, Friendship, Hate, Love     | Companionship       |
|5.       | Five, Seven, Thousand, Three              | Thousand            |
|---------|-------------------------------------------|---------------------|

I made a game out of this which can be played in the console where you can see if you can guess which word the AI picked. Below is a link to the GitHub project. The file called *create_new_pairs.py* contains the code for the discussed model.

[Link to the GitHub project](https://github.com/erikbrorson/one_odd_out_game)