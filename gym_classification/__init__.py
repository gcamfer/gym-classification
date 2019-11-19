from gym.envs.registration import register

register(
    id='RLClassification-v0',
    entry_point='gym_classification.envs.env_4_RL_classification:Env4RLClassification',
    #kwargs={X:None,y:None,batch_size:None,output_shape:None,randomize:False,custom_rewards:None}
)
