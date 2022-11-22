import torch


def generate_images_and_stack_features(accelerator, fmgan, eval_model, eval_fake_dl, quantize):
    feature_holder, prob_holder, image_holder = [], [], []
    
    for idx, data in enumerate(eval_fake_dl):
        with torch.inference_mode():
            p1_hat = fmgan.generator(data['p1'], data['r1'])
            # p1_hat_gather = accelerator.gather_for_metrics((p1_hat))

            features, logits = eval_model.get_outputs(p1_hat, quantize=quantize)
            probs = torch.nn.functional.softmax(logits, dim=1)

        feature_holder.append(features)
        prob_holder.append(probs)
        image_holder.append(p1_hat)


    feature_holder = torch.cat(feature_holder, 0)
    prob_holder = torch.cat(prob_holder, 0)
    image_holder = torch.cat(image_holder, 0)
    
    feature_holder = accelerator.gather_for_metrics(feature_holder)
    prob_holder = accelerator.gather_for_metrics(prob_holder)
    image_holder = accelerator.gather_for_metrics(image_holder)
    # feature_holder = torch.cat(losses.GatherLayer.apply(feature_holder), dim=0)
    # prob_holder = torch.cat(losses.GatherLayer.apply(prob_holder), dim=0)
    # [5000, 2048], [5000, 1008], [5000, 3, 256, 256]
    return feature_holder, prob_holder, image_holder
    