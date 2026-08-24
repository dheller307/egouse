from dataset_api import Hot3dDataProvider
from data_loaders.loader_object_library import load_object_library
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId

RGB_STREAM_ID = StreamId("214-1")


def init_provider(sequence_path, object_library_path):
  object_library = load_object_library(object_library_folderpath=object_library_path)
  hot3d_data_provider = Hot3dDataProvider(
    sequence_folder=sequence_path,
    object_library=object_library,
    mano_hand_model=None,
  )
  device_data_provider = hot3d_data_provider.device_data_provider
  timestamps = device_data_provider.get_sequence_timestamps()
  hand_provider = hot3d_data_provider.hand_box2d_data_provider
  object_provider = hot3d_data_provider.object_box2d_data_provider
  return (
    hot3d_data_provider,
    device_data_provider,
    timestamps,
    hand_provider,
    object_provider,
    object_library,
  )


def get_hand_boxes(timestamp_ns, hand_provider):
  hand_boxes_with_dt = hand_provider.get_bbox_at_timestamp(
    stream_id=RGB_STREAM_ID,
    timestamp_ns=timestamp_ns,
    time_query_options=TimeQueryOptions.CLOSEST,
    time_domain=TimeDomain.TIME_CODE,
  )
  if hand_boxes_with_dt is None:
    return None
  boxes = hand_boxes_with_dt.box2d_collection.box2ds
  if boxes is None:
    return None
  return {"boxes": boxes, "time_dt": hand_boxes_with_dt.time_delta_ns}


def get_object_boxes(timestamp_ns, object_provider):
  object_boxes_with_dt = object_provider.get_bbox_at_timestamp(
    stream_id=RGB_STREAM_ID,
    timestamp_ns=timestamp_ns,
    time_query_options=TimeQueryOptions.CLOSEST,
    time_domain=TimeDomain.TIME_CODE,
  )
  if object_boxes_with_dt is None:
    return None
  boxes = object_boxes_with_dt.box2d_collection.box2ds
  if boxes is None:
    return None
  return {"boxes": boxes, "time_dt": object_boxes_with_dt.time_delta_ns}
