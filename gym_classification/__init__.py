from gym.envs.registration import register

register(
    id='RLClassification-v0',
    entry_point='gym_classification.envs:Env4RLClassification',
)
