import collections
import json
import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from domainbed.mmd import mmd_loss
from domainbed.datasets import get_dataset, split_dataset
from domainbed import algorithms
from domainbed.evaluator import Evaluator
from domainbed.lib import misc
from domainbed.lib import swa_utils
from domainbed.lib.query import Q
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed import swad as swad_module

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def json_handler(v):
    if isinstance(v, (Path, range)):
        return str(v)
    raise TypeError(f"`{type(v)}` is not JSON Serializable")


def train(test_envs, args, hparams, n_steps, checkpoint_freq, logger, writer, target_env=None):
    logger.info("")

    #######################################################
    # setup dataset & loader
    #######################################################
    args.real_test_envs = test_envs  # for log
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    dataset, in_splits, out_splits = get_dataset(test_envs, args, hparams, algorithm_class)
    test_splits = []
    if hparams.indomain_test > 0.0:
        logger.info("!!! In-domain test mode On !!!")
        assert hparams["val_augment"] is False, (
            "indomain_test split the val set into val/test sets. "
            "Therefore, the val set should be not augmented."
        )
        val_splits = []
        for env_i, (out_split, _weights) in enumerate(out_splits):
            n = len(out_split) // 2
            seed = misc.seed_hash(args.trial_seed, env_i)
            val_split, test_split = split_dataset(out_split, n, seed=seed)
            val_splits.append((val_split, None))
            test_splits.append((test_split, None))
            logger.info(
                "env %d: out (#%d) -> val (#%d) / test (#%d)"
                % (env_i, len(out_split), len(val_split), len(test_split))
            )
        out_splits = val_splits

    if target_env is not None:
        testenv_name = f"te_{dataset.environments[target_env]}"
        logger.info(f"Target env = {target_env}")
    else:
        testenv_properties = [str(dataset.environments[i]) for i in test_envs]
        testenv_name = "te_" + "_".join(testenv_properties)

    logger.info(
        "Testenv name escaping {} -> {}".format(testenv_name, testenv_name.replace(".", ""))
    )
    testenv_name = testenv_name.replace(".", "")
    logger.info(f"Test envs = {test_envs}, name = {testenv_name}")

    n_envs = len(dataset)
    train_envs = sorted(set(range(n_envs)) - set(test_envs))
    iterator = misc.SplitIterator(test_envs)
    batch_sizes = np.full([n_envs], hparams["batch_size"], dtype=np.int)

    batch_sizes[test_envs] = 0
    batch_sizes = batch_sizes.tolist()

    logger.info(f"Batch sizes for each domain: {batch_sizes} (total={sum(batch_sizes)})")

    # calculate steps per epoch
    steps_per_epochs = [
        len(env) / batch_size
        for (env, _), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ]
    steps_per_epoch = min(steps_per_epochs)
    # epoch is computed by steps_per_epoch
    prt_steps = ", ".join([f"{step:.2f}" for step in steps_per_epochs])
    logger.info(f"steps-per-epoch for each domain: {prt_steps} -> min = {steps_per_epoch:.2f}")



    # setup loaders
    train_loaders = [
        InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=batch_size,
            num_workers=dataset.N_WORKERS,
        )
        for (env, env_weights), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ]

    # setup eval loaders
    eval_loaders_kwargs = []
    for i, (env, _) in enumerate(in_splits + out_splits + test_splits):
        batchsize = hparams["test_batchsize"]
        loader_kwargs = {"dataset": env, "batch_size": batchsize, "num_workers": dataset.N_WORKERS}
        if args.prebuild_loader:
            loader_kwargs = FastDataLoader(**loader_kwargs)
        eval_loaders_kwargs.append(loader_kwargs)

    eval_weights = [None for _, weights in (in_splits + out_splits + test_splits)]
    eval_loader_names = ["env{}_in".format(i) for i in range(len(in_splits))]
    eval_loader_names += ["env{}_out".format(i) for i in range(len(out_splits))]
    eval_loader_names += ["env{}_inTE".format(i) for i in range(len(test_splits))]
    eval_meta = list(zip(eval_loader_names, eval_loaders_kwargs, eval_weights))

    #######################################################
    # setup algorithm (model)
    #######################################################
    hparams["total_num_domains"] = len(dataset)
    algorithm = algorithm_class(
        dataset.input_shape,
        dataset.num_classes,
        len(dataset) - len(test_envs),
        hparams,
    )
    num_class = dataset.num_classes
    algorithm.to(device)

    
    n_params = sum([p.numel() for p in algorithm.parameters()])
    logger.info("# of params = %d" % n_params)

    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    #######################################################
    # start training loop
    #######################################################
    evaluator = Evaluator(
        test_envs,
        eval_meta,
        n_envs,
        logger,
        evalmode=args.evalmode,
        debug=args.debug,
        target_env=target_env,
    )

    swad = None
    if hparams["swad"]:
        swad_algorithm = swa_utils.AveragedModel(algorithm)
        swad_cls = getattr(swad_module, hparams["swad"])
        swad = swad_cls(evaluator, **hparams.swad_kwargs)

    last_results_keys = None
    records = []
    epochs_path = args.out_dir / "results.jsonl"
    algorithm.re_w = False
    # Visualize
    # with torch.no_grad():
    #     embedding_train = []
    #     y_train = []
    #     for train_env in train_envs:
    #         domain_data = in_splits[train_env][0]
    #         domain_dataset = torch.utils.data.DataLoader(domain_data, batch_size=max(batch_sizes)*4, num_workers=dataset.N_WORKERS)
    #         embedding_domain = []
    #         y_domain = []
    #         for batch in domain_dataset:
    #             batch = {
    #                             key: tensor.to(device) for key, tensor in batch.items()
    #                         }
    #             embedding_batch = algorithm.predict_embed(batch["x"])
    #             y_batch = batch["y"]
    #             y_domain.append(y_batch)
    #             embedding_domain.append(embedding_batch)
    #         embedding_domain = torch.cat(embedding_domain, 0)
    #         y_domain = torch.cat(y_domain, 0)
    #         embedding_train.append(embedding_domain)
    #         y_train.append(y_domain)
    #     embedding_train = torch.cat(embedding_train, 0)
    #     y_train = torch.cat(y_train, 0)
    #     embedding_test = []
    #     y_test = []
    #     for test_env in test_envs:
    #         domain_data = in_splits[test_env][0]
    #         domain_dataset = torch.utils.data.DataLoader(domain_data, batch_size=max(batch_sizes)*4, num_workers=dataset.N_WORKERS)
    #         embedding_domain = []
    #         y_domain = []
    #         for batch in domain_dataset:
    #             batch = {
    #                             key: tensor.to(device) for key, tensor in batch.items()
    #                         }
    #             embedding_batch = algorithm.predict_embed(batch["x"])
    #             y_batch = batch["y"]
    #             y_domain.append(y_batch)
    #             embedding_domain.append(embedding_batch)
    #         embedding_domain = torch.cat(embedding_domain, 0)
    #         y_domain = torch.cat(y_domain, 0)
    #         embedding_test = embedding_domain
    #         y_test = y_domain
    #     embedding_train, embedding_test, y_train, y_test = embedding_train.detach().cpu().numpy(), embedding_test.detach().cpu().numpy(), y_train.detach().cpu().numpy(), y_test.detach().cpu().numpy()
    #     np.save("vis/pacs_embedding_train.npy", embedding_train)
    #     np.save("vis/pacs_embedding_test.npy", embedding_test)
    #     np.save("vis/pacs_y_train.npy", y_train)
    #     np.save("vis/pacs_y_test.npy", y_test)

    for step in range(n_steps):

        # weight calculation

        # if args.re_w and step==hparams["start_epoch"]:
        #     algorithm.re_w = True
        #     embedding_all = []
        #     y_all = []
        #     with torch.no_grad():
        #         for train_env in train_envs:
        #             domain_data = in_splits[train_env][0]
        #             domain_dataset = torch.utils.data.DataLoader(domain_data, batch_size=max(batch_sizes)*4, num_workers=dataset.N_WORKERS)
        #             embedding_domain = []
        #             y_domain = []
        #             for batch in domain_dataset:
        #                 batch = {
        #                                 key: tensor.to(device) for key, tensor in batch.items()
        #                             }
        #                 embedding_batch = algorithm.predict_embed(batch["x"])
        #                 y_batch = batch["y"]
        #                 y_domain.append(y_batch)
        #                 embedding_domain.append(embedding_batch)
        #             embedding_domain = torch.cat(embedding_domain, 0)
        #             y_domain = torch.cat(y_domain, 0)
        #             y_all.append(y_domain)
        #             embedding_all.append(embedding_domain)
        #         # weight_matrix = np.zeros((n_envs,n_envs))
        #         # for i, train_env_1 in enumerate(train_envs):
        #         #     for j, train_env_2 in enumerate(train_envs):
        #         #         if j < i+1:
        #         #             continue
        #         #         dis_i_j = mmd_loss(embedding_all[i], embedding_all[j])
        #         #         weight_matrix[train_env_1][train_env_2] = dis_i_j
        #         #         weight_matrix[train_env_2][train_env_1] = dis_i_j

        #         # plus label information
        #         embedding_y_domain_all = []
        #         for i, y_domain in enumerate(y_all):
        #             for label in range(num_class):
        #                 label_index = torch.where(y_domain==label)[0]
        #                 # some are empty
        #                 embedding_y_domain = embedding_all[i][label_index]
        #                 embedding_y_domain_all.append(embedding_y_domain)
        #         weight_matrix = np.ones((n_envs*num_class,n_envs*num_class))
        #         tau = 1
        #         for i, train_env_1 in enumerate(train_envs):
        #             for label_1 in range(num_class):
        #                 for j, train_env_2 in enumerate(train_envs):
        #                     for label_2 in range(num_class):
        #                         index_1 = num_class*train_env_1+label_1
        #                         index_2 = num_class*train_env_2+label_2
        #                         if index_2 < index_1+1:
        #                             continue
        #                         # if label_1==label_2:
        #                         #     # set weight to zero for the same label
        #                         #     continue
        #                         embedding_1, embedding_2 = embedding_y_domain_all[num_class*i+label_1], embedding_y_domain_all[num_class*j+label_2]
        #                         if embedding_1.shape[0]==0 or embedding_2.shape[0]==0:
        #                             continue
        #                         dis = mmd_loss(embedding_1, embedding_2)
        #                         dis = torch.exp(-dis/tau)
        #                         weight_matrix[index_1][index_2] = dis
        #                         weight_matrix[index_2][index_1] = dis
        #     algorithm.weight_matrix = torch.tensor(weight_matrix).cuda()
        
        step_start_time = time.time()
        # batches_dictlist: [{env0_data_key: tensor, env0_...}, env1_..., ...]
        batches_dictlist = next(train_minibatches_iterator)
        # batches: {data_key: [env0_tensor, ...], ...}
        batches = misc.merge_dictlist(batches_dictlist)
        # to device
        batches = {
            key: [tensor.to(device) for tensor in tensorlist] for key, tensorlist in batches.items()
        }

        inputs = {**batches, "step": step}
        step_vals = algorithm.update(**inputs)
        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)
        checkpoint_vals["step_time"].append(time.time() - step_start_time)
        if args.log: 
            with open("loss_analysis.txt","a+") as f:
                f.write("step {}:".format(step)+str(step_vals["ce_loss"])+" "+str(step_vals["sup_cl_loss"])+" "+str(step_vals["pre_cl_loss"])+"\n")
        if swad:
            # swad_algorithm is segment_swa for swad
            swad_algorithm.update_parameters(algorithm, step=step)

        # Visualize
        # if step==200:
        #     with torch.no_grad():
        #         embedding_train = []
        #         y_train = []
        #         for train_env in train_envs:
        #             domain_data = in_splits[train_env][0]
        #             domain_dataset = torch.utils.data.DataLoader(domain_data, batch_size=max(batch_sizes)*4, num_workers=dataset.N_WORKERS)
        #             embedding_domain = []
        #             y_domain = []
        #             for batch in domain_dataset:
        #                 batch = {
        #                                 key: tensor.to(device) for key, tensor in batch.items()
        #                             }
        #                 embedding_batch = algorithm.predict_embed(batch["x"])
        #                 y_batch = batch["y"]
        #                 y_domain.append(y_batch)
        #                 embedding_domain.append(embedding_batch)
        #             embedding_domain = torch.cat(embedding_domain, 0)
        #             y_domain = torch.cat(y_domain, 0)
        #             embedding_train.append(embedding_domain)
        #             y_train.append(y_domain)
        #         embedding_train = torch.cat(embedding_train, 0)
        #         y_train = torch.cat(y_train, 0)
        #         embedding_test = []
        #         y_test = []
        #         for test_env in test_envs:
        #             domain_data = in_splits[test_env][0]
        #             domain_dataset = torch.utils.data.DataLoader(domain_data, batch_size=max(batch_sizes)*4, num_workers=dataset.N_WORKERS)
        #             embedding_domain = []
        #             y_domain = []
        #             for batch in domain_dataset:
        #                 batch = {
        #                                 key: tensor.to(device) for key, tensor in batch.items()
        #                             }
        #                 embedding_batch = algorithm.predict_embed(batch["x"])
        #                 y_batch = batch["y"]
        #                 y_domain.append(y_batch)
        #                 embedding_domain.append(embedding_batch)
        #             embedding_domain = torch.cat(embedding_domain, 0)
        #             y_domain = torch.cat(y_domain, 0)
        #             embedding_test = embedding_domain
        #             y_test = y_domain
        #         embedding_train, embedding_test, y_train, y_test = embedding_train.detach().cpu().numpy(), embedding_test.detach().cpu().numpy(), y_train.detach().cpu().numpy(), y_test.detach().cpu().numpy()
        #         np.save("vis/pacs_cl_embedding_train.npy", embedding_train)
        #         np.save("vis/pacs_cl_embedding_test.npy", embedding_test)
        #         np.save("vis/pacs_cl_y_train.npy", y_train)
        #         np.save("vis/pacs_cl_y_test.npy", y_test)
        #         exit()

        if step % checkpoint_freq == 0:
            results = {
                "step": step,
                "epoch": step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            eval_start_time = time.time()
            accuracies, summaries = evaluator.evaluate(algorithm)
            results["eval_time"] = time.time() - eval_start_time

            # results = (epochs, loss, step, step_time)
            results_keys = list(summaries.keys()) + sorted(accuracies.keys()) + list(results.keys())
            # merge results
            results.update(summaries)
            results.update(accuracies)

            # print
            if results_keys != last_results_keys:
                logger.info(misc.to_row(results_keys))
                last_results_keys = results_keys
            logger.info(misc.to_row([results[key] for key in results_keys]))
            records.append(copy.deepcopy(results))

            # update results to record
            results.update({"hparams": dict(hparams), "args": vars(args)})

            with open(epochs_path, "a") as f:
                f.write(json.dumps(results, sort_keys=True, default=json_handler) + "\n")

            checkpoint_vals = collections.defaultdict(lambda: [])

            writer.add_scalars_with_prefix(summaries, step, f"{testenv_name}/summary/")
            writer.add_scalars_with_prefix(accuracies, step, f"{testenv_name}/all/")

            if args.model_save and step >= args.model_save:
                ckpt_dir = args.out_dir / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)

                test_env_str = ",".join(map(str, test_envs))
                filename = "TE{}_{}.pth".format(test_env_str, step)
                if len(test_envs) > 1 and target_env is not None:
                    train_env_str = ",".join(map(str, train_envs))
                    filename = f"TE{target_env}_TR{train_env_str}_{step}.pth"
                path = ckpt_dir / filename

                save_dict = {
                    "args": vars(args),
                    "model_hparams": dict(hparams),
                    "test_envs": test_envs,
                    "model_dict": algorithm.cpu().state_dict(),
                }
                algorithm.cuda()
                if not args.debug:
                    torch.save(save_dict, path)
                else:
                    logger.debug("DEBUG Mode -> no save (org path: %s)" % path)

            # swad
            if swad:
                def prt_results_fn(results, avgmodel):
                    step_str = f" [{avgmodel.start_step}-{avgmodel.end_step}]"
                    row = misc.to_row([results[key] for key in results_keys if key in results])
                    logger.info(row + step_str)

                swad.update_and_evaluate(
                    swad_algorithm, results["train_out"], results["tr_outloss"], prt_results_fn
                )

                if hasattr(swad, "dead_valley") and swad.dead_valley:
                    logger.info("SWAD valley is dead -> early stop !")
                    break

                swad_algorithm = swa_utils.AveragedModel(algorithm)  # reset

        if step % args.tb_freq == 0:
            # add step values only for tb log
            writer.add_scalars_with_prefix(step_vals, step, f"{testenv_name}/summary/")
    # final prediction

    # visualize
    # with torch.no_grad():
    #     embedding_train = []
    #     y_train = []
    #     for train_env in train_envs:
    #         domain_data = in_splits[train_env][0]
    #         domain_dataset = torch.utils.data.DataLoader(domain_data, batch_size=max(batch_sizes)*4, num_workers=dataset.N_WORKERS)
    #         embedding_domain = []
    #         y_domain = []
    #         for batch in domain_dataset:
    #             batch = {
    #                             key: tensor.to(device) for key, tensor in batch.items()
    #                         }
    #             embedding_batch = algorithm.predict_embed(batch["x"])
    #             y_batch = batch["y"]
    #             y_domain.append(y_batch)
    #             embedding_domain.append(embedding_batch)
    #         embedding_domain = torch.cat(embedding_domain, 0)
    #         y_domain = torch.cat(y_domain, 0)
    #         embedding_train.append(embedding_domain)
    #         y_train.append(y_domain)
    #     embedding_train = torch.cat(embedding_train, 0)
    #     y_train = torch.cat(y_train, 0)
    #     embedding_test = []
    #     y_test = []
    #     for test_env in test_envs:
    #         domain_data = in_splits[test_env][0]
    #         domain_dataset = torch.utils.data.DataLoader(domain_data, batch_size=max(batch_sizes)*4, num_workers=dataset.N_WORKERS)
    #         embedding_domain = []
    #         y_domain = []
    #         for batch in domain_dataset:
    #             batch = {
    #                             key: tensor.to(device) for key, tensor in batch.items()
    #                         }
    #             embedding_batch = algorithm.predict_embed(batch["x"])
    #             y_batch = batch["y"]
    #             y_domain.append(y_batch)
    #             embedding_domain.append(embedding_batch)
    #         embedding_domain = torch.cat(embedding_domain, 0)
    #         y_domain = torch.cat(y_domain, 0)
    #         embedding_test = embedding_domain
    #         y_test = y_domain
    #     embedding_train, embedding_test, y_train, y_test = embedding_train.detach().cpu().numpy(), embedding_test.detach().cpu().numpy(), y_train.detach().cpu().numpy(), y_test.detach().cpu().numpy()
    #     np.save("vis/pacs_erm_embedding_train.npy", embedding_train)
    #     np.save("vis/pacs_erm_embedding_test.npy", embedding_test)
    #     np.save("vis/pacs_erm_y_train.npy", y_train)
    #     np.save("vis/pacs_erm_y_test.npy", y_test)
    #     exit()

    # find best
    logger.info("---")
    records = Q(records)
    oracle_best = records.argmax("test_out")["test_in"]
    iid_best = records.argmax("train_out")["test_in"]
    last = records[-1]["test_in"]

    if hparams.indomain_test:
        # if test set exist, use test set for indomain results
        in_key = "train_inTE"
    else:
        in_key = "train_out"

    iid_best_indomain = records.argmax("train_out")[in_key]
    last_indomain = records[-1][in_key]

    ret = {
        "oracle": oracle_best,
        "iid": iid_best,
        "last": last,
        "last (inD)": last_indomain,
        "iid (inD)": iid_best_indomain,
    }

    # Evaluate SWAD
    if swad:
        swad_algorithm = swad.get_final_model()
        if hparams["freeze_bn"] is False:
            n_steps = 500 if not args.debug else 10
            logger.warning(f"Update SWAD BN statistics for {n_steps} steps ...")
            swa_utils.update_bn(train_minibatches_iterator, swad_algorithm, n_steps)

        logger.warning("Evaluate SWAD ...")
        accuracies, summaries = evaluator.evaluate(swad_algorithm)
        results = {**summaries, **accuracies}
        start = swad_algorithm.start_step
        end = swad_algorithm.end_step
        step_str = f" [{start}-{end}]  (N={swad_algorithm.n_averaged})"
        row = misc.to_row([results[key] for key in results_keys if key in results]) + step_str
        logger.info(row)

        ret["SWAD"] = results["test_in"]
        ret["SWAD (inD)"] = results[in_key]

        # with torch.no_grad():
        #     embedding_train = []
        #     y_train = []
        #     for train_env in train_envs:
        #         domain_data = in_splits[train_env][0]
        #         domain_dataset = torch.utils.data.DataLoader(domain_data, batch_size=max(batch_sizes)*4, num_workers=dataset.N_WORKERS)
        #         embedding_domain = []
        #         y_domain = []
        #         for batch in domain_dataset:
        #             batch = {
        #                             key: tensor.to(device) for key, tensor in batch.items()
        #                         }
        #             embedding_batch = swad_algorithm.predict_embed(batch["x"])
        #             y_batch = batch["y"]
        #             y_domain.append(y_batch)
        #             embedding_domain.append(embedding_batch)
        #         embedding_domain = torch.cat(embedding_domain, 0)
        #         y_domain = torch.cat(y_domain, 0)
        #         embedding_train.append(embedding_domain)
        #         y_train.append(y_domain)
        #     embedding_train = torch.cat(embedding_train, 0)
        #     y_train = torch.cat(y_train, 0)
        #     embedding_test = []
        #     y_test = []
        #     for test_env in test_envs:
        #         domain_data = in_splits[test_env][0]
        #         domain_dataset = torch.utils.data.DataLoader(domain_data, batch_size=max(batch_sizes)*4, num_workers=dataset.N_WORKERS)
        #         embedding_domain = []
        #         y_domain = []
        #         for batch in domain_dataset:
        #             batch = {
        #                             key: tensor.to(device) for key, tensor in batch.items()
        #                         }
        #             embedding_batch = swad_algorithm.predict_embed(batch["x"])
        #             y_batch = batch["y"]
        #             y_domain.append(y_batch)
        #             embedding_domain.append(embedding_batch)
        #         embedding_domain = torch.cat(embedding_domain, 0)
        #         y_domain = torch.cat(y_domain, 0)
        #         embedding_test = embedding_domain
        #         y_test = y_domain
        #     embedding_train, embedding_test, y_train, y_test = embedding_train.detach().cpu().numpy(), embedding_test.detach().cpu().numpy(), y_train.detach().cpu().numpy(), y_test.detach().cpu().numpy()
        #     np.save("vis/pacs_swad_embedding_train.npy", embedding_train)
        #     np.save("vis/pacs_swad_embedding_test.npy", embedding_test)
        #     np.save("vis/pacs_swad_y_train.npy", y_train)
        #     np.save("vis/pacs_swad_y_test.npy", y_test)
        #     exit()

    for k, acc in ret.items():
        logger.info(f"{k} = {acc:.3%}")

    return ret, records
