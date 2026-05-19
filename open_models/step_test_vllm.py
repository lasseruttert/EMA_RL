import json, os, torch

if __name__ == '__main__':
    from trl import GRPOConfig
    from rl.reward import OpenAIGraderReward
    from validate import TrainingConfig
    from grpo_steer_trl import (load_grpo_dataset, load_model_hf, SteeredLoRASyncGRPOTrainer,
                                 SteeringHook, load_steering_vectors)

    CONFIG = os.environ["STEP_TEST_CONFIG"]

    with open(CONFIG) as f:
        cfg = TrainingConfig(**json.load(f))

    print('loading model via load_model_hf...')
    model, tokenizer = load_model_hf(cfg.model, load_in_4bit=cfg.load_in_4bit,
                                      max_seq_length=cfg.max_seq_length)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    dataset = load_grpo_dataset(cfg.training_file, grader_type=cfg.grader_type, include_answer=True)
    reward_fn = OpenAIGraderReward(model=cfg.reward_model, grader_type=cfg.grader_type,
        print_training=cfg.print_training).reward_function

    training_args = GRPOConfig(
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_seq_length - cfg.max_prompt_length,
        use_vllm=False, temperature=cfg.rl_temperature,
        learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay, warmup_ratio=0.1,
        lr_scheduler_type=cfg.lr_scheduler_type, optim=cfg.optim, logging_steps=1,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_generations=cfg.num_generations, num_train_epochs=1,
        max_steps=1,
        report_to='none', output_dir=cfg.output_dir, save_strategy='no',
        beta=cfg.beta, max_grad_norm=cfg.max_grad_norm,
    )

    intervention_dict = load_steering_vectors(cfg.steering_config)
    steering_hooks = {k: SteeringHook(v, alpha=float(cfg.steering_config.get('steering_coef', 1.0)))
                      for k, v in intervention_dict.items()}

    print(f'Using SteeredLoRASyncGRPOTrainer, vllm={cfg.vllm_base_model}, gpu_util={cfg.vllm_gpu_util}')
    trainer = SteeredLoRASyncGRPOTrainer(
        steering_hooks=steering_hooks,
        vllm_base_model=cfg.vllm_base_model,
        vllm_gpu_util=float(cfg.vllm_gpu_util),
        max_lora_rank=cfg.r,
        model=model, processing_class=tokenizer,
        reward_funcs=[reward_fn], args=training_args, train_dataset=dataset,
    )
    print('trainer init OK — running 1 step...')
    trainer.train()
    print('=== 1 step completed successfully ===')
