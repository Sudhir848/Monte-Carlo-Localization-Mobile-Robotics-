from grid import *
from particle import Particle
from utils import *
from setting import *
import numpy as np
import math

# ------------------------------------------------------------------------
def motion_update(particles, odom):
    """ Particle filter motion update

        Arguments: 
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- noisy odometry measurements, a pair of robot pose, i.e. last time
                step pose and current time step pose

        Returns: the list of particle representing belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """
    oldPosition, newPosition = odom
    alpha1 = 0.001
    alpha2 = 0.001
    alpha3 = 0.001
    alpha4 = 0.001

    rot1 = math.degrees(math.atan2(newPosition[1] - oldPosition[1], newPosition[0] - oldPosition[0])) - oldPosition[2]
    trans = np.sqrt((newPosition[0] - oldPosition[0])**2 + (newPosition[1] - oldPosition[1])**2)
    rot2 = newPosition[2] - oldPosition[2] - rot1

    newParticles = []
    for particle in particles:
        newRot1 = rot1 + np.random.normal(0, alpha1 * abs(rot1) + alpha2 * trans)
        newTrans = trans + np.random.normal(0, alpha3 * trans + alpha4 * (rot1 + rot2))
        newRot2 = rot2 + np.random.normal(0, alpha1 * abs(rot2) + alpha2 * trans)

        new_x = particle.x + newTrans * math.cos(math.radians(particle.h + newRot1))
        new_y = particle.y + newTrans * math.sin(math.radians(particle.h + newRot1))
        new_h = (particle.h + newRot1 + newRot2) % 360
        newParticles.append(Particle(new_x + np.random.normal(0, ODOM_TRANS_SIGMA), new_y + np.random.normal(0, ODOM_TRANS_SIGMA), new_h + np.random.normal(0, ODOM_HEAD_SIGMA)))
    return newParticles

# ------------------------------------------------------------------------
def measurement_update(particles, measured_marker_list, grid):
    """ Particle filter measurement update

        Arguments: 
        particles -- a list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before measurement update
        measured_marker_list -- robot detected marker list, each marker has format:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree
        grid -- grid world map containing the marker information. 
                see grid.py and CozGrid for definition

        Returns: the list of particle representing belief p(x_{t} | u_{t})
                after measurement update
    """
    if not measured_marker_list:
        return [Particle(p.x, p.y, p.h) for p in particles]

    weightArray = []
    for particle in particles:
        visibleMarkers = particle.read_markers(grid)
        if visibleMarkers:
            weights = []
            for measuredMarker in measured_marker_list:
                measuredMarkerX, measuredMarkerY, _ = measuredMarker
                nearestMarker = None
                minimum_distance = float('inf')
                for marker in visibleMarkers:
                    visibleMarkerX, visibleMarkerY, _ = marker
                    distance = np.sqrt((measuredMarkerX - visibleMarkerX)**2 + (measuredMarkerY - visibleMarkerY)**2)
                    if distance < minimum_distance:
                        minimum_distance = distance
                        nearestMarker = marker

                if nearestMarker:
                    vMX, vMY, vMH = nearestMarker
                    distance = minimum_distance
                    diff_heading = abs(vMH - measuredMarker[2]) % 360
                    prob = math.exp(-0.5 * ((distance / MARKER_TRANS_SIGMA)**2 + (diff_heading / MARKER_ROT_SIGMA)**2))
                    weights.append(prob if prob > 0 else 1e-100)
            weight = np.prod(weights)
        else:
            weight = 1e-100  # Minimal weight if no markers visible

        weightArray.append((particle, weight))

    total_weight = sum(w for _, w in weightArray)
    if total_weight > 0:
        normalized_weights = [w / total_weight for _, w in weightArray]
    else:
        normalized_weights = [1.0 / len(particles)] * len(particles)

    chosen_particles = np.random.choice([p for p, _ in weightArray], size=len(particles), p=normalized_weights, replace=True)
    return [Particle(p.x, p.y, p.h) for p in chosen_particles]