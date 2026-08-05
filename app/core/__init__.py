# =============================================================================
# core — UI から切り離したロジック層
#
#   画面（Streamlit のウィジェット）を含まない処理をここに置く。
#   main.py からは `from core import *` で読み込む。
#
#   依存の向き（循環させないこと）:
#     config → state → utils → provenance → dataset → その他
#   config は他のどのモジュールにも依存しない。
# =============================================================================
from __future__ import annotations

from .config import *          # noqa: F401,F403
from .state import *           # noqa: F401,F403
from .utils import *           # noqa: F401,F403
from .provenance import *      # noqa: F401,F403
from .cvat import *            # noqa: F401,F403
from .dataset import *         # noqa: F401,F403
from .models import *          # noqa: F401,F403
from .training import *        # noqa: F401,F403
from .evaluation import *      # noqa: F401,F403
from .inference import *       # noqa: F401,F403
from .serverless import *      # noqa: F401,F403
from .fiftyone_app import *    # noqa: F401,F403
from .errors import *          # noqa: F401,F403
from .augment_preview import *  # noqa: F401,F403
from .experiments import *     # noqa: F401,F403
from .extensions import *      # noqa: F401,F403
from .mosaic import *          # noqa: F401,F403
from .crop import *            # noqa: F401,F403
from .cleanup import *         # noqa: F401,F403
from .model_prefs import *     # noqa: F401,F403
from .hostpath import *        # noqa: F401,F403
from .review import *          # noqa: F401,F403

# `import *` はアンダースコア始まりを取り込まないため、明示的に公開する
from .config import (  # noqa: F401
    _DOC_AUG, _DOC_TRAIN, _MODEL_OPTS,
)
from .cvat import (  # noqa: F401
    _collect_prediction_items,
)
from .dataset import (  # noqa: F401
    _yolo_txt_to_xyxy,
)
from .evaluation import (  # noqa: F401
    _eval_worker,
)
from .inference import (  # noqa: F401
    _draw_predictions,
)
from .serverless import (  # noqa: F401
    _deploy_worker, _nuctl,
)
from .state import (  # noqa: F401
    _get_deploy_shared, _get_eval_shared, _get_train_shared,
)
from .training import (  # noqa: F401
    _StdoutCapture, _train_worker,
)
from .utils import (  # noqa: F401
    _box_iou, _find_image_dirs, _iou,
)
