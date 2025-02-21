import pickle
import functools
from typing import List, Tuple

from accelerate import init_empty_weights
import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

from model import ChameleonXLLMXConfig, ChameleonXLLMXForConditionalGenerationAnyRes
from xllmx.data.item_processor import ItemProcessorBase
from xllmx.solvers.finetune import FinetuneSolverBase

from transformers import AutoProcessor, AutoModel
from PIL import Image

from xllmx.model.tokenizer import Tokenizer
import xllmx.util as util
import xllmx.util.lr_sched as lr_sched
import xllmx.util.misc as misc
from xllmx.util.tensor_type import promote_param_to_fp32

from tools import process_images

class ItemProcessor(ItemProcessorBase):
    def __init__(self):
        super().__init__()

        vit_root = "./ckpts/SigLIP"
        self.vit_processor = AutoProcessor.from_pretrained(vit_root)

        # AnyRes configuration
        self.anyres_cfg = {}
        image_grid_pinpoints = {
            "image_grid_pinpoints": [
                [384, 768],
                [768, 384],
                [768, 768],
                [384, 1152],
                [1152, 384]
            ]}
        image_aspect_ratio = {
            "image_aspect_ratio": "anyres"
        }
        self.anyres_cfg.update(**image_grid_pinpoints)
        self.anyres_cfg.update(**image_aspect_ratio)
        

    def process_item(self, data_item: dict, training_mode=False) -> Tuple[List, List]:
        assert training_mode

        if "token" in data_item and "label" in data_item:
            data_item = data_item
        else:
            assert "file" in data_item
            file_path = data_item["file"]
            with open(data_item["file"], "rb") as f:
                data_item = pickle.load(f)

        tokens = data_item["token"]
        labels = data_item["label"]
        assert len(tokens) == len(labels)

        # Process Continuous VIT features
        if len(data_item['raw_image']) == 0:   # For text-only data, where no image paths are provided
            image_tensor = torch.zeros(1,3,384,384)
            return (tokens, image_tensor, (0,0)), labels
        try:
            image_path = data_item['raw_image'][0]
            image = Image.open(image_path).convert("RGB")
            image_tensor = process_images([image], self.vit_processor.image_processor, self.anyres_cfg)[0]
        except:
            image = Image.new('RGB', (256, 256), (0, 0, 0))
            none_labels = [-100] * len(labels)
            labels = none_labels
            print(f'{image_path} is NONE!')
            image_tensor = self.vit_processor(images=image, return_tensors="pt").pixel_values   # (1,3,384,384)

        return (tokens, image_tensor, image.size), labels

    def predict_item_token_length(self, data_item: dict) -> int:
        if "token" in data_item:
            return len(data_item["token"])
        elif "len" in data_item:
            return data_item["len"]
        else:
            raise ValueError()


class Solver(FinetuneSolverBase):
    @classmethod
    def get_args_parser(cls):
        parser = super().get_args_parser()
        # task-specific parameters
        parser.add_argument("--max_seq_len", default=4096, type=int, help="max token length")
        parser.add_argument("--mask_image_logits", default=True)
        parser.add_argument("--unmask_image_logits", action="store_false", dest="mask_image_logits")
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--z_loss_weight", type=float, default=0.0)
        parser.add_argument("--model_size", type=str, default="7B")
        # The whole model consists of 'model'(ChameleonModel), 'lm_head', 'vit'(SiglipVisionTransformer), 'adapter'
        parser.add_argument("--trainable_params", type=str, default="adapter")
        parser.add_argument("--vit_lr_scale", type=float, default=0.0)
        parser.add_argument("--stage", type=str, default="I")    
        return parser

    def _model_func(
        self,
        init_from: str,
    ) -> (ChameleonXLLMXForConditionalGenerationAnyRes, None):

        # Only instantiate the model on rank0
        # Other ranks will receive the model weights from rank0 during FSDP wrapping (through `sync_module_states`)
        # See https://github.com/pytorch/pytorch/issues/105840
        if self.dp_rank == 0:
            model = ChameleonXLLMXForConditionalGenerationAnyRes.from_pretrained(
                init_from,
                max_position_embeddings=self.args.max_seq_len,
                mask_image_logits=self.args.mask_image_logits,
                dropout=self.args.dropout,
                z_loss_weight=self.args.z_loss_weight,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
            )
            if self.args.stage == "I":
                self.logger.info("Start Stage I Training ...")
                self.logger.info("Init External ViT...")
                model._init_vit()
            elif self.args.stage == "II":
                self.logger.info("Start Stage II Training ...")
                pass # ViT has been saved in Stage I ckpt
            else:
                raise ValueError(f"Unrecognized training stage {self.args.stage}")
        else:
            with init_empty_weights():
                config = ChameleonXLLMXConfig.from_pretrained(
                    init_from,
                    max_position_embeddings=self.args.max_seq_len,
                    mask_image_logits=self.args.mask_image_logits,
                    dropout=self.args.dropout,
                    z_loss_weight=self.args.z_loss_weight,
                    torch_dtype=torch.bfloat16,
                )
                model = ChameleonXLLMXForConditionalGenerationAnyRes(config)
            if self.args.stage == "I":
                self.logger.info("Start Stage I Training ...")
                self.logger.info("Init External ViT...")
                model._init_vit()
            elif self.args.stage == "II":
                self.logger.info("Start Stage II Training ...")
                pass # ViT has been saved in Stage I ckpt
            else:
                raise ValueError(f"Unrecognized training stage {self.args.stage}")
        del model.model.vqmodel
        
        return model, None

    def _item_processor_func(self) -> ItemProcessorBase:
        return ItemProcessor()

    def _make_and_save_starting_point(self, save_path: str) -> None:

        pretrained_name = {
             # "7B": "Alpha-VLLM/Chameleon_7B_mGPT",
            "7B": "./ckpts/Lumina-mGPT-7B-512",
            "34B": "Alpha-VLLM/Chameleon_34B_mGPT",
        }[self.args.model_size]

        model = ChameleonXLLMXForConditionalGenerationAnyRes.from_pretrained(
            pretrained_name,
            max_position_embeddings=self.args.max_seq_len,
            mask_image_logits=self.args.mask_image_logits,
            dropout=self.args.dropout,
            z_loss_weight=self.args.z_loss_weight,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )

        image_tokens = model.model.vocabulary_mapping.image_tokens
        model.lm_head.weight.data[image_tokens] = torch.zeros_like(model.lm_head.weight.data[image_tokens])

        model.save_pretrained(save_path, max_shard_size="10GB")

    def build_model(self) -> (nn.Module, Tokenizer):
        init_from = self.args.resume_path or self.args.init_from
        if init_from is None:
            starting_point_path = Path(self.args.output_dir) / "starting_point"
            if dist.get_rank() == 0:
                if (starting_point_path / "config.json").exists():
                    self.logger.info(f"will use existing starting point at {starting_point_path}")
                    self.logger.info(
                        f"***********************************************************************\n"
                        f"********************************Caution********************************\n"
                        f"Caution: the starting point is created by some previous experiment run \n"
                        f"If the starting point saved by that run is broken, or if the expected  \n"
                        f"starting weights for the model has changed since that run, please manu-\n"
                        f"remove the saved path: \n"
                        f"{starting_point_path} \n"
                        f"and rerun the experiment.\n"
                        f"***********************************************************************\n"
                        f"***********************************************************************\n"
                    )
                else:
                    self.logger.info(f"creating starting-point weights at {starting_point_path}")
                    self._make_and_save_starting_point(save_path=str(starting_point_path))
            dist.barrier()
            init_from = str(starting_point_path)

        self.logger.info(f"Start instantiating unwrapped model from {init_from}")

        # only rank 0 instantiate, otherwise to meta
        unwrapped_model, tokenizer = self._model_func(init_from)

        ## Force freeze vit `patch_embedding`, `position_embedding`
        force_frozen_params = ["vit.embeddings.patch_embedding", "vit.embeddings.position_embedding"]
        # Rewrite traianble parameter setup logic
        if hasattr(unwrapped_model, "get_trainable_params"):
            # trainable_params = dict(unwrapped_model.get_trainable_params())
            trainable_params = unwrapped_model.get_trainable_params(self.args.trainable_params)
            for key, param in unwrapped_model.named_parameters():
                if any([_ in key for _ in trainable_params]):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    # keep_fp32_keywords = ["norm", "lm_head", "embed_tokens"]
                    # if any([_ in key for _ in keep_fp32_keywords]):
                    #     promote_param_to_fp32(param)
                    # elif param.is_floating_point():
                    #     param.data = param.data.to(self.mixed_precision_dtype)
                if any([_ in key for _ in force_frozen_params]):
                    param.requires_grad = False
                promote_param_to_fp32(param)
                # # Force all paramters to be set as bf16 for debug usage
                # param.data = param.data.to(torch.bfloat16)
        else:
            self.logger.warning(
                f"model class {type(unwrapped_model)} does not have `get_trainable_params` method,"
                f"set all params to trainable"
            )
            for key, param in unwrapped_model.named_parameters():
                param.requires_grad = True
                param.requires_grad = True
                promote_param_to_fp32(param)

        self.logger.info("Finish instantiating unwrapped model.")
        self.logger.info(f"Unwrapped model: \n{str(unwrapped_model)}")
        self.logger.info(f"Model config: \n{unwrapped_model.config.to_dict()}")

        # ----------------
        self.is_peft = getattr(unwrapped_model, "is_peft", False)  # todo
        self.logger.info(f"Model is Peft: {self.is_peft}")
        # ----------------

        misc.mark_mp_params(unwrapped_model)

        # defer this after FSDP
        misc.print_param_status(unwrapped_model)

        train_param_count_local, train_param_count_all = 0, 0
        frozen_param_count_local, frozen_param_count_all = 0, 0
        for name, param in unwrapped_model.named_parameters():
            model_parallel = getattr(param, "model_parallel", False)
            if param.requires_grad:
                if model_parallel:
                    train_param_count_all += param.numel() * fs_init.get_model_parallel_world_size()
                else:
                    train_param_count_all += param.numel()
                train_param_count_local += param.numel()
            else:
                if model_parallel:
                    frozen_param_count_all += param.numel() * fs_init.get_model_parallel_world_size()
                else:
                    frozen_param_count_all += param.numel()
                frozen_param_count_local += param.numel()

        self.logger.info(
            f"Trainable parameter count : {train_param_count_local} (local rank), {train_param_count_all} (all).\n"
            f"Frozen parameter count : {frozen_param_count_local} (local rank), {frozen_param_count_all} (all)."
        )

        # checkpointing (part1, should be called before FSDP wrapping)
        if self.args.checkpointing:
            # todo more hints for not-implemented
            checkpointing_list = unwrapped_model.get_checkpointing_wrap_module_list()
        else:
            checkpointing_list = []

        # todo pre-sync ignored states
        model = self.setup_fsdp_sync(
            unwrapped_model, self.args.data_parallel, self.args.precision, self.args.grad_precision
        )

        # broadcast non-model-parallel parameters within model parallel group
        misc.broadcast_nonmp_parameters(model)

        # checkpointing (part2, after FSDP wrapping)
        if self.args.checkpointing:
            print("apply gradient checkpointing")
            non_reentrant_wrapper = functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=lambda submodule: submodule in checkpointing_list,
            )

        self.logger.info(f"Wrapped model: \n{str(model)}")

        # Setup optimizer
        param_lr_mapping = [
            {'params': [p for n, p in model.named_parameters() if "vit" in n], 'lr_scale': self.args.vit_lr_scale},       # 对 model.vit 层使用单独的学习率 specific_lr
            {'params': [p for n, p in model.named_parameters() if "vit" not in n]}  # 其余参数使用默认学习率 self.args.lr
        ]
        opt = torch.optim.AdamW(param_lr_mapping, lr=self.args.lr, weight_decay=self.args.wd, betas=(0.9, 0.95))


        return model, tokenizer, opt


if __name__ == "__main__":
    args = Solver.get_args_parser().parse_args()
    solver = Solver(args)
    solver.run()
