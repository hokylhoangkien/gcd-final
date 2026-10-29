"""Checked in-memory adapter: no writes to upstream; all edits fail closed.

Changes: runtime paths, DINO-v1-only import, final-epoch policy, epoch resume hooks.
The baseline loss and geometry remain untouched. Not an empirical parity claim.
"""
import ast
import hashlib
from gcd_runtime.config import TRAIN_SHA


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Upstream anchor changed: {old[:70]}')
    return source.replace(old, new, 1)


def prepare_source(source):
    if hashlib.sha256(source.encode()).hexdigest() != TRAIN_SHA:
        raise ValueError('Unreviewed upstream training source')
    source = replace_once(source, 'from models import vision_transformer2 as vits2',
                          '# DINOv2 is deliberately unsupported by this v1 adapter')
    source = replace_once(source, 'from kmeans_pytorch import kmeans',
                          'from gcd_runtime.compat import kmeans')
    source = replace_once(source, 'for epoch in range(args.epochs):', '''start_epoch = _runtime.restore(model, projection_head, optimizer, exp_lr_scheduler)
    if start_epoch >= args.epochs:
        raise ValueError('Checkpoint already reached final epoch; use evaluation instead')
    for epoch in range(start_epoch, args.epochs):''')
    source = replace_once(source, 'if old_acc_test > best_test_acc_lab:',
                          'if False:  # runtime uses final epoch, never test-selected best')
    source = replace_once(source, "    print('############# Final Reports #############')", '''        _runtime.save(epoch, model, projection_head, optimizer, exp_lr_scheduler,
                      all_acc, old_acc, new_acc, all_acc_test, old_acc_test, new_acc_test)

    best_stats = [all_acc, old_acc, new_acc] if start_epoch < args.epochs else [float('nan')] * 3
    print('############# Final Reports #############')''')
    source = replace_once(source, '        return metrics\n', '        _runtime.evaluation(metrics)\n        return metrics\n')
    source = replace_once(source, "print('Best Train Epochs:  {}'.format(best_epoch))",
                          "print('Checkpoint policy: final_epoch; completed epoch {}'.format(epoch))")
    source = replace_once(source, "print('Best Train Epochs:  {} '.format(best_epoch))",
                          "print('Final epoch: {}'.format(args.epochs - 1))")
    source = replace_once(source, "print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(best_stats[0], best_stats[1], best_stats[2]))",
                          "print('Final reported train metrics: All {:.4f} | Old {:.4f} | New {:.4f}'.format(best_stats[0], best_stats[1], best_stats[2]))")
    source = replace_once(source, '            loss_angle, loss_distance = 0.0, 0.0',
                          '            _runtime.begin_batch(epoch, batch_idx)\n            loss_angle, loss_distance = 0.0, 0.0')
    source = replace_once(source,
                          '            sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)\n            sup_con_loss_angle = sup_con_crit_angle(sup_con_feats, labels=sup_con_labels)',
                          '            sup_con_loss, sup_con_loss_angle = _runtime.supervised_losses(sup_con_crit, sup_con_crit_angle, sup_con_feats, sup_con_labels)')
    source = replace_once(source,
                          '                sup_con_loss += sup_con_crit(semisup_con_feats[:, :, :int(dimension / 2 ** (i + 1))], labels=preds_ind_list[i][np.argsort(uq_index)[uq_idxs]]) / 2 ** (i + 1)\n                sup_con_loss_angle += sup_con_crit_angle(semisup_con_feats[:, :, :int(dimension / 2 ** (i + 1))], labels=preds_ind_list[i][np.argsort(uq_index)[uq_idxs]]) / 2 ** (i + 1)',
                          '                sup_con_loss += _runtime.hierarchy_loss(sup_con_crit, semisup_con_feats[:, :, :int(dimension / 2 ** (i + 1))], preds_ind_list[i][np.argsort(uq_index)[uq_idxs]], mask_lab, uq_idxs, epoch, i) / 2 ** (i + 1)\n                sup_con_loss_angle += _runtime.hierarchy_loss(sup_con_crit_angle, semisup_con_feats[:, :, :int(dimension / 2 ** (i + 1))], preds_ind_list[i][np.argsort(uq_index)[uq_idxs]], mask_lab, uq_idxs, epoch, i) / 2 ** (i + 1)')
    source = replace_once(source,
                          '            confusion_factor = confusion_factor / confusion_factor.sum(dim=1)',
                          '            confusion_factor = _runtime.normalize_confusion(confusion_factor)')
    source = replace_once(source, '            loss = loss_in',
                          '            loss = _runtime.check_loss(loss_in, epoch, batch_idx)')
    ast.parse(source)
    return source
