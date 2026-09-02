# Mapping and navigation

## How does the AR-1 navigate?

A spinning lidar turret on top measures distance to walls and furniture and
builds a 2D map. Cliff sensors on the underside stop it falling down stairs. A
bumper handles the obstacles the lidar cannot see, such as glass table legs and
dark matte skirting.

## How many maps can the AR-1 store?

Up to three saved maps, so a three-storey house works. Carry the robot to the
floor you want, start a clean, and it matches the live scan against its saved
maps. Matching fails if the furniture has moved a lot, in which case it builds
a new map and you should delete the stale one.

## How do I set a no-go zone?

In the app, open the map, tap Zones, then Add no-go zone, and drag the
rectangle over the area. No-go zones are enforced by the robot, not the app, so
they still apply if your phone is off.

## Why does the AR-1 miss the same corner every time?

Usually the lidar cannot see into it. Deep alcoves, dark furniture below the
lidar plane and chair legs in a tight cluster all read as a solid block. Move
one chair or add a spot clean for that corner.

## Does the AR-1 work in the dark?

Yes. The lidar is an active sensor and does not need room light. The optional
camera-based obstacle avoidance on the AR-1 Pro does need light, and it falls
back to lidar-only navigation in a dark room.

## Why did the AR-1 clean rooms in a different order?

Route order is recalculated each run from the robot's start position and the
remaining battery. It is not a fault. Fix the order by cleaning rooms
individually from the app if it matters.
