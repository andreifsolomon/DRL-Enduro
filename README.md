# OPENAI GYM Enduro

### Overview

nduro consists of maneuvering a race car in the National Enduro, a long-distance endurance race. The object of the race is to pass a certain number of cars each day. Doing so will allow the player to continue racing for the next day. The driver must avoid other racers and pass 200 cars on the first day, and 300 cars with each following day.

As the time in the game passes, visibility changes as well. When it is night in the game the player can only see the oncoming cars’ taillights. As the days progress, cars will become more difficult to avoid as well. Weather and time of day are factors in how to play. During the day the player may drive through an icy patch on the road which would limit control of the vehicle, or a patch of fog may reduce visibility.

Description from [Wikipedia](https://en.wikipedia.org/wiki/Enduro_%28video_game%29)


### Introduction

This repository contains an implementation of **Duel Deep Q-Network (DQN)** agent running in a [OPENAI GYM Enduro](https://gym.openai.com/envs/Enduro-ram-v0) Environment that can be used to train and evaluate models.

Maximize your score in the Atari 2600 game Enduro. In this environment, the observation is the RAM of the Atari machine, consisting of (only!) 128 bytes.
