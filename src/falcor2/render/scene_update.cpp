// SPDX-License-Identifier: Apache-2.0

#include "scene_update.h"

#include <sgl/device/device.h>
#include <sgl/device/command.h>

namespace falcor {

SceneUpdateContext::SceneUpdateContext(sgl::Device* device)
    : m_device(device)
{
}

sgl::CommandEncoder* SceneUpdateContext::command_encoder()
{
    if (!m_command_encoder)
        m_command_encoder = m_device->create_command_encoder();
    return m_command_encoder;
}

void SceneUpdateContext::submit()
{
    if (m_command_encoder) {
        m_device->submit_command_buffer(m_command_encoder->finish());
        m_command_encoder = nullptr;
    }
}

} // namespace falcor
