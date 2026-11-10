"""Autonomous "Reset Scene" patrol mission.

A :class:`ResetMission` drives the robot to every object in the cache to find
which have gone missing, re-scans for objects that have appeared at new
locations, and — for objects that both went missing *and* reappeared — drives to
the new location and publishes the object's point cloud.

The mission logic here is deliberately decoupled from ROS and viser (MIT 6.031:
minimize coupling, program to an interface): it depends only on the small set of
callables injected at construction, so the algorithm can be edited independently
of the wiring in ``main_ros2.py``.
"""

from typing import Callable, List, Optional, Tuple
import time


class ResetMission:
    """A three-phase "what changed since last time?" patrol.

    Phase 1 — drive to every cached object; at each, force a fresh view and run
      ``check_missing_object`` to accumulate the set of objects that have gone.
    Phase 2 — run ``check_added_object`` to find objects at new locations.
    Phase 3 — for every object that both went missing and reappeared (matched by
      query label, the only common key between the two results): drive to the new
      location, publish its point cloud, and wait for the manipulator to pick it
      up; then drive to where it used to be, publish the drop-off location, and
      wait for the manipulator to set it back down.

    Parameters
    ----------
    objects_module :
        The ``Objects`` instance holding the object cache and the
        ``check_missing_object`` / ``check_added_object`` operations.
    solver :
        The VGGT-SLAM solver, passed straight through to the ``check_*`` calls.
    navigate_to : Callable[[Detection], bool]
        Drive to ``det.bbox.center`` and block until arrival; returns True on
        arrival, False if the object is unreachable or the mission was aborted.
    refresh_view : Callable[[], None]
        Force a fresh submap at the current pose so ``check_missing_object`` sees
        what the robot sees now.
    capture_object : Callable[[Detection], None]
        Publish ``det``'s point cloud in the current camera frame.
    place_object : Callable[[Detection], None], optional
        Publish ``det``'s bbox-centroid drop-off location in the current camera
        frame (the "Place Object" action).  Used in phase 3 to tell the
        manipulator where to set the object back down.  Defaults to no-op.
    back_up : Callable[[], bool], optional
        Reverse the robot straight out of an object (after a pick-up / drop-off)
        so it leaves without a rotate-and-drive maneuver that could graze the
        obstacle.  Returns True on completion, False if aborted.  Defaults to a
        no-op that returns True.
    with_data_lock :
        A context manager (the pipeline's ``data_lock``) held only around the
        ``check_*`` inference calls, mirroring the manual viewer buttons.
    wait_for_pickup : Callable[[], bool], optional
        Called after ``capture_object`` to block until an external manipulator
        reports the object was picked up.  Returns True once picked up, False if
        the mission was aborted while waiting.  Defaults to not waiting (returns
        True immediately).
    wait_for_dropoff : Callable[[], bool], optional
        Called after ``place_object`` to block until the manipulator reports the
        object was set back down.  Same contract as ``wait_for_pickup``.
    set_status : Callable[[str], None], optional
        Progress reporter (e.g. writes a viser markdown). Defaults to no-op.
    should_abort : Callable[[], bool], optional
        Polled between steps; when it returns True the mission stops promptly.
        Defaults to never aborting.
    """

    def __init__(
        self,
        objects_module,
        solver,
        *,
        navigate_to: Callable[[object], bool],
        refresh_view: Callable[[], None],
        capture_object: Callable[[object], None],
        with_data_lock,
        place_object: Optional[Callable[[object], None]] = None,
        back_up: Optional[Callable[[], bool]] = None,
        wait_for_pickup: Optional[Callable[[], bool]] = None,
        wait_for_dropoff: Optional[Callable[[], bool]] = None,
        set_status: Optional[Callable[[str], None]] = None,
        should_abort: Optional[Callable[[], bool]] = None,
    ):
        self._objects = objects_module
        self._solver = solver
        self._navigate_to = navigate_to
        self._refresh_view = refresh_view
        self._capture_object = capture_object
        self._place_object = place_object or (lambda _det: None)
        self._back_up = back_up or (lambda: True)
        self._with_data_lock = with_data_lock
        self._wait_for_pickup = wait_for_pickup or (lambda: True)
        self._wait_for_dropoff = wait_for_dropoff or (lambda: True)
        self._set_status = set_status or (lambda _msg: None)
        self._should_abort = should_abort or (lambda: False)

    def run(self) -> None:
        """Execute the full patrol. Safe to run in a background thread."""
        # Snapshot the cache BEFORE phase 2: check_added_object appends its new
        # detections to the cache, so capturing (query, detection) pairs up front
        # keeps phase 1 iterating over only the originally-cached objects.
        snapshot: List[Tuple[str, object]] = [
            (query, det)
            for query, detections in self._objects.cached_memory.items()
            for det in detections
            if det is not None
        ]
        if not snapshot:
            self._set_status("Reset: object cache is empty — nothing to patrol.")
            return

        # Original (pre-move) location per query, so phase 3 can return an object
        # to where it used to be.  First cached detection per query (mirrors the
        # query-level granularity of the missing/added logic below).
        original_by_query: dict = {}
        for query, det in snapshot:
            original_by_query.setdefault(query, det)

        # --- Phase 1: patrol cached objects, accumulate the missing set --------
        missing: set = set()
        for query, det in snapshot:
            if self._should_abort():
                self._set_status("Reset aborted.")
                return
            # Carry the running missing list so it stays visible during the drive
            # (a bare per-object message would be overwritten too fast to read).
            tail = f"  (missing so far: {', '.join(sorted(missing))})" if missing else ""
            self._set_status(f"Visiting '{query}'…{tail}")
            if not self._navigate_to(det):
                continue  # unreachable or aborted mid-drive — skip
            self._refresh_view()
            with self._with_data_lock:
                newly_missing = self._objects.check_missing_object(self._solver)
            missing.update(newly_missing)
            # Report this object's outcome to viser.
            if query in newly_missing:
                self._set_status(f"'{query}' detected MISSING.")
            else:
                self._set_status(f"'{query}' still present.")
            time.sleep(3) # Give this user a chance to see the viser printout message.

            # Back straight out before turning to the next object, so we don't
            # rotate-and-graze the obstacle this object sits against.
            self._set_status(f"Backing away from '{query}'…")
            if not self._back_up():
                self._set_status("Reset aborted.")
                return

        # --- Phase 2: scan for objects that appeared at new locations ----------
        if self._should_abort():
            self._set_status("Reset aborted.")
            return
        self._set_status("Scanning for new objects…")
        with self._with_data_lock:
            added = self._objects.check_added_object(self._solver)  # {query: [Detection]}

        # --- Phase 3: recapture objects that both went missing and reappeared --
        targets: List[Tuple[str, object]] = [
            (query, det)
            for query, detections in added.items()
            if query in missing
            for det in detections
        ]
        for query, det in targets:
            if self._should_abort():
                self._set_status("Reset aborted.")
                return
            self._set_status(f"Recapturing moved '{query}'…")
            if not self._navigate_to(det):
                continue  # unreachable or aborted mid-drive — skip
            self._capture_object(det)
            # Wait for the external manipulator to pick the object up before
            # moving on, so we don't disturb the scene mid-grasp.
            self._set_status(f"Published '{query}' — waiting for pickup…")
            if not self._wait_for_pickup():
                self._set_status("Reset aborted.")
                return
            self._set_status(f"'{query}' picked up.")

            # Back straight out before turning, so we don't graze the obstacle on
            # the way out (the approach was normal to the surface; the exit
            # retraces it in reverse).
            self._set_status(f"Backing away from '{query}'…")
            if not self._back_up():
                self._set_status("Reset aborted.")
                return

            # Return the object to where it used to be: drive to the original
            # cached location, publish the drop-off point, wait for confirmation.
            orig_det = original_by_query.get(query)
            if orig_det is None:
                continue  # no known original location — leave it held
            self._set_status(f"Returning '{query}' to its original spot…")
            if not self._navigate_to(orig_det):
                continue  # original spot unreachable or aborted — skip
            self._place_object(orig_det)
            self._set_status(f"Placing '{query}' — waiting for dropoff…")
            if not self._wait_for_dropoff():
                self._set_status("Reset aborted.")
                return
            self._set_status(f"'{query}' returned to its original spot.")

            # Back straight out of the drop-off spot before moving to the next.
            self._set_status(f"Backing away from '{query}' drop-off…")
            if not self._back_up():
                self._set_status("Reset aborted.")
                return

        missing_str = ", ".join(sorted(missing)) if missing else "none"
        self._set_status(
            f"Reset complete — missing: {missing_str}; {len(targets)} recaptured."
        )
