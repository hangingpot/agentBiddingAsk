---
name: opsx-edit
description: 编辑 OpenSpec 规范、提案、设计或任务文件。Use when the user invokes /opsx-edit, /opsx:edit, asks to modify OpenSpec specs, or wants to adjust proposal.md, design.md, tasks.md, or openspec/specs files.
disable-model-invocation: true
---

# Opsx Edit

用于修改既有 OpenSpec 文件，而不是创建新变更或实现代码。

## 输入

用户应说明要修改的变更、能力、文件或具体需求。若缺少目标或修改意图，先询问；不要猜测写入哪个规范。

## 工作流

1. **定位目标**
   - 若用户给出变更名，优先编辑 `openspec/changes/<change>/` 下的产出物。
   - 若用户给出能力名，编辑 `openspec/specs/<capability>/spec.md` 或对应变更下的 `specs/<capability>/spec.md`。
   - 若当前上下文中有明确打开的 OpenSpec 文件，可把它作为候选目标，但在修改前说明将编辑哪个文件。
   - 默认不要编辑 `openspec/changes/archive/`；除非用户明确要求修改归档内容。

2. **读取上下文**
   - 修改前读取目标文件。
   - 若编辑变更内的 spec，同时读取同一变更的 `proposal.md`、`design.md`、`tasks.md` 中已存在且相关的文件。
   - 若修改基线规范，必要时搜索相关 active changes，避免和未归档变更冲突。

3. **编辑规则**
   - 保持已有标题、语言、术语和粒度风格。
   - 规范需求使用“必须/禁止”等规范性措辞。
   - 每个需求至少包含一个 `#### 场景`，场景使用“当/那么”结构并可验证。
   - 修改既有能力时优先更新完整需求块，不要只补零散句子。
   - `tasks.md` 使用 `- [ ]` 或 `- [x]` 复选框，不改变无关任务状态。
   - `proposal.md` 保持“为什么 / 变更内容 / 功能 (Capabilities) / 影响”结构。
   - 不把 CLI 输出中的 `context`、`rules`、内部说明或聊天过程写进产出物。

4. **质量检查**
   - 修改后快速复读目标片段，确认没有重复标题、断裂列表或空需求。
   - 如果可用，运行与变更相关的 OpenSpec 校验命令，例如：
     ```bash
     openspec-cn validate <change-name> --strict
     ```
   - 校验失败时修复规范问题；若失败原因超出本次修改范围，向用户说明。

## 输出

完成后简要说明：

- 修改了哪些 OpenSpec 文件
- 关键规范变化
- 是否运行校验，以及结果

不要在总结中粘贴大段规范全文，除非用户要求。
