import torch
def grad_cam_1d(model,x,target_class=None):
    acts=[]; grads=[]
    h1=model.block3.register_forward_hook(lambda m,i,o:acts.append(o))
    h2=model.block3.register_full_backward_hook(lambda m,gi,go:grads.append(go[0]))
    model.eval(); model.zero_grad(set_to_none=True); logits=model(x)
    cls=int(logits.argmax(1).item()) if target_class is None else int(target_class); logits[0,cls].backward()
    a=acts[0].detach()[0]; g=grads[0].detach()[0]; w=g.mean(1,keepdim=True)
    cam=torch.relu((w*a).sum(0)); cam=cam/(cam.max()+1e-8)
    cam=torch.nn.functional.interpolate(cam[None,None],size=x.shape[-1],mode="linear",align_corners=False)[0,0]
    h1.remove(); h2.remove(); return cam.detach().cpu().numpy()
