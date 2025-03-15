"""题目求解模块"""

from typing import Generator, Tuple, Optional, Any, List, Dict
from openai import OpenAI

from backend.config.settings import settings
from backend.logger.log_config import logger
from backend.core.utils import convert_formula_format
from backend.core.image_processor import image_processor
from prompts.solver_prompts import (
    SOLVER_SYSTEM_PROMPT,
    get_solver_prompt,
    PROBLEM_BREAKDOWN_SYSTEM_PROMPT,
    get_breakdown_prompt
)

class ProblemSolver:
    """题目求解类"""
    def __init__(self):
        """初始化OpenAI客户端"""
        self.client = OpenAI(
            base_url=settings.api_base_url,
            api_key=settings.api_key,
        )
        
    def _break_down_problem(
        self,
        text_input: str,
        image_description: Optional[str] = None,
        is_complex_mode: bool = False
    ) -> Generator[Tuple[str, str], None, None]:
        """
        将问题拆分为多个子问题
        
        Args:
            text_input: 题目文本
            image_description: 图片描述（可选）
            is_complex_mode: 是否使用复杂模式
            
        Yields:
            Tuple[str, str]: (拆分结果, 日志内容)
        """
        # 获取模型
        _, solver_model = settings.get_model_info(is_complex_mode)
        
        # 构建提示词
        full_problem = get_breakdown_prompt(text_input, image_description)
        
        # 构建消息
        messages = [
            {
                "role": "system",
                "content": PROBLEM_BREAKDOWN_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": full_problem
            }
        ]
        
        try:
            # 创建请求
            stream = self.client.chat.completions.create(
                model=solver_model,
                messages=messages,
                stream=True,
                temperature=0.01
            )
            
            # 流式接收并更新输出
            collected_chunks = []
            buffer = []
            
            for chunk in stream:
                if not (hasattr(chunk, 'choices') and
                       chunk.choices and
                       hasattr(chunk.choices[0], 'delta') and
                       hasattr(chunk.choices[0].delta, 'content') and
                       chunk.choices[0].delta.content is not None):
                    continue
                    
                content = chunk.choices[0].delta.content
                collected_chunks.append(content)
                buffer.append(content)
                
                # 输出当前缓冲区内容
                if buffer:
                    current_content = "".join(buffer)
                    yield current_content, ""
            
            if not collected_chunks:
                raise Exception("未收到模型响应")
                
            # 生成最终输出和日志
            final_content = "".join(buffer)
            log_str = logger.log_api_interaction(solver_model, messages, final_content)
            logger.logger.info(f"{solver_model} 问题拆分完成")
            
        except Exception as e:
            error_msg = f"问题拆分出错：{str(e)}"
            logger.log_error(error_msg)
            yield error_msg, ""
            
    def _parse_subproblems(self, breakdown_result: str) -> List[Dict[str, str]]:
        """
        解析拆分后的子问题
        
        Args:
            breakdown_result: 拆分结果
            
        Returns:
            List[Dict[str, str]]: 子问题列表，每个子问题包含 'id', 'title', 'content' 字段
        """
        subproblems = []
        lines = breakdown_result.split('\n')
        current_problem = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 检查是否是子问题标题（格式如：## 子问题1: 标题 或 ## 子问题X: [原题中的第X个问题]）
            if line.startswith('## 子问题') and ':' in line:
                if current_problem:
                    subproblems.append(current_problem)
                
                parts = line.split(':', 1)
                problem_id = parts[0].strip('# ').strip()
                title = parts[1].strip() if len(parts) > 1 else ""
                
                # 提取题号（如：子问题1 -> 1）
                problem_number = ''.join(filter(str.isdigit, problem_id))
                
                current_problem = {
                    'id': problem_id,
                    'number': problem_number,  # 添加题号字段
                    'title': title,
                    'content': ''
                }
            elif current_problem is not None:
                current_problem['content'] += line + '\n'
        
        # 添加最后一个子问题
        if current_problem:
            subproblems.append(current_problem)
            
        # 按题号排序
        if subproblems and all('number' in p for p in subproblems):
            try:
                subproblems.sort(key=lambda p: int(p['number']) if p['number'].isdigit() else float('inf'))
            except (ValueError, TypeError):
                # 如果排序失败，保持原顺序
                pass
            
        return subproblems

    def _update_output(
        self,
        content: str,
        current_output: list,
        replace_last: bool = False,
        add_separator: bool = True
    ) -> Generator[Tuple[str, str], None, None]:
        """
        更新输出内容
        
        Args:
            content: 新内容
            current_output: 当前输出列表
            replace_last: 是否替换最后一个内容
            add_separator: 是否添加分隔符
            
        Yields:
            Tuple[str, str]: (输出内容, 日志内容)
        """
        # 如果是替换最后一个内容，先移除
        if replace_last and current_output:
            current_output.pop()
        
        # 添加新内容
        content_with_separator = content + ("\n\n---\n\n" if add_separator else "")
        current_output.append(content_with_separator)
        
        # 只显示实际内容，不显示API调用记录
        yield "\n\n".join(current_output).rstrip('---\n\n'), ""

    def solve_problem(
        self,
        text_input: str,
        image: Optional[Any] = None,
        is_complex_mode: bool = False
    ) -> Generator[Tuple[str, str], None, None]:
        """
        处理完整题目求解流程
        
        Args:
            text_input: 题目文本
            image: 题目图片（可选）
            is_complex_mode: 是否使用复杂模式
            
        Yields:
            Tuple[str, str]: (解答内容, 日志内容)
        """
        api_logs = []
        full_result = []
        current_output = []
        image_description = None
        
        # 处理图片描述
        if image is not None:
            try:
                description_gen = image_processor.get_image_description(
                    text_input, image, is_complex_mode
                )
                latest_desc = []
                
                # 处理流式输出
                for desc in description_gen:
                    if isinstance(desc, tuple):  # 如果是最终结果
                        final_desc, _ = desc
                        if "出错" in final_desc:  # 如果是错误信息
                            api_logs.append(f"# API调用错误\n{final_desc}")
                            yield final_desc, "\n\n".join(api_logs)
                            return
                        latest_desc = [final_desc]
                    else:  # 流式输出的部分内容
                        latest_desc = [desc]
                        yield from self._update_output(
                            f"# 图片描述\n\n{desc}",
                            current_output,
                            replace_last=True
                        )
                
                if latest_desc:
                    image_description = latest_desc[0]
                    full_result.append(image_description)
                    logger.logger.info("图片描述完成")
                else:
                    raise Exception("未获取到图片描述")
                    
            except Exception as e:
                error_msg = f"图片处理出错：{str(e)}"
                logger.log_error(error_msg)
                yield error_msg, ""
                return
        
        # 获取求解器模型
        _, solver_model = settings.get_model_info(is_complex_mode)
        
        # 步骤1: 拆分问题为子问题
        yield from self._update_output(
            f"# 问题拆分\n\n正在按题号将问题拆分为子问题...",
            current_output,
            add_separator=True
        )
        
        # 收集完整的拆分结果，但不输出中间过程
        breakdown_result = ""
        for result, _ in self._break_down_problem(text_input, image_description, is_complex_mode):
            breakdown_result = result
            # 不再每次都输出中间结果
        
        # 只在拆分完成后输出一次最终结果
        if breakdown_result:
            yield from self._update_output(
                f"# 问题拆分\n\n{breakdown_result}",
                current_output,
                replace_last=True
            )
        
        # 解析子问题
        subproblems = self._parse_subproblems(breakdown_result)
        if not subproblems:
            error_msg = "无法按题号拆分问题为子问题，请检查问题格式"
            logger.log_error(error_msg)
            yield from self._update_output(
                f"# 问题拆分出错\n\n{error_msg}",
                current_output,
                add_separator=True
            )
            return
            
        # 步骤2: 逐个解决子问题
        previous_solutions = []  # 存储前面子问题的解答结果
        
        for i, subproblem in enumerate(subproblems):
            subproblem_id = subproblem['id']
            subproblem_title = subproblem['title']
            subproblem_content = subproblem['content']
            
            yield from self._update_output(
                f"# {subproblem_id}: {subproblem_title}\n\n正在求解...",
                current_output,
                add_separator=True
            )
            
            # 构建子问题描述
            sub_problem_text = f"{subproblem_title}\n\n{subproblem_content}"
            
            # 添加前面子问题的解答结果作为上下文
            context = ""
            if previous_solutions and i > 0:
                context = "\n\n# 前面小问的解答结果\n\n"
                for j, prev_solution in enumerate(previous_solutions):
                    prev_id = subproblems[j]['id']
                    prev_title = subproblems[j]['title']
                    context += f"## {prev_id}: {prev_title}\n{prev_solution}\n\n"
            
            # 构建完整问题描述
            full_problem = get_solver_prompt(sub_problem_text, image_description)
            if context:
                full_problem += "\n---\n" + context
            
            # 构建消息
            messages = [
                {
                    "role": "system",
                    "content": SOLVER_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": full_problem
                }
            ]
            
            try:
                # 根据模式决定是否添加 reasoning_effort
                if is_complex_mode and solver_model == "o3-mini":
                    stream = self.client.chat.completions.create(
                        model=solver_model,
                        messages=messages,
                        reasoning_effort="high",
                        stream=True,
                        temperature=0.01
                    )
                else:
                    stream = self.client.chat.completions.create(
                        model=solver_model,
                        messages=messages,
                        stream=True,
                        temperature=0.01
                    )
                    
                # 流式接收并更新输出
                collected_chunks = []
                buffer = []
                formula_buffer = []
                in_formula = False
                latex_start = ""  # 记录LaTeX公式的开始标记
                
                try:
                    for chunk in stream:
                        if not (hasattr(chunk, 'choices') and
                               chunk.choices and
                               hasattr(chunk.choices[0], 'delta') and
                               hasattr(chunk.choices[0].delta, 'content') and
                               chunk.choices[0].delta.content is not None):
                            continue
                            
                        content = chunk.choices[0].delta.content
                        collected_chunks.append(content)
                        
                        # 处理LaTeX公式
                        for char in content:
                            if not in_formula:
                                if char == '\\':
                                    latex_start = char
                                    continue
                                elif latex_start:
                                    latex_start += char
                                    if latex_start == "\\[" or latex_start == "\\(":
                                        in_formula = True
                                        formula_buffer = []
                                        buffer.append("$$" if latex_start == "\\[" else "$")
                                        latex_start = ""
                                    elif len(latex_start) > 1:
                                        buffer.extend(list(latex_start))
                                        latex_start = ""
                                else:
                                    buffer.append(char)
                            else:
                                formula_buffer.append(char)
                                if (len(formula_buffer) >= 2 and
                                    formula_buffer[-2] == '\\' and
                                    formula_buffer[-1] == ']'):
                                    in_formula = False
                                    buffer.extend(formula_buffer[:-2])
                                    buffer.append("$$")
                                    formula_buffer = []
                                elif (len(formula_buffer) >= 2 and
                                      formula_buffer[-2] == '\\' and
                                      formula_buffer[-1] == ')'):
                                    in_formula = False
                                    buffer.extend(formula_buffer[:-2])
                                    buffer.append("$")
                                    formula_buffer = []
                        
                        # 输出当前缓冲区内容
                        if buffer:
                            current_content = f"# {subproblem_id}: {subproblem_title}\n\n{''.join(buffer)}"
                            yield from self._update_output(
                                current_content,
                                current_output,
                                replace_last=True
                            )
                    
                    if not collected_chunks:
                        raise Exception("未收到模型响应")
                        
                    # 生成最终输出和日志
                    solution_content = ''.join(buffer)
                    final_content = f"# {subproblem_id}: {subproblem_title}\n\n{solution_content}"
                    log_str = logger.log_api_interaction(solver_model, messages, final_content)
                    logger.logger.info(f"{solver_model} 子问题 {i+1}/{len(subproblems)} 求解完成")
                    
                    # 存储当前子问题的解答结果，供后续子问题使用
                    previous_solutions.append(solution_content)
                    
                except Exception as e:
                    error_msg = f"模型响应处理出错：{str(e)}"
                    logger.log_error(f"Stream处理错误 - 模型: {solver_model}, 错误: {str(e)}")
                    yield from self._update_output(
                        f"# {subproblem_id}: {subproblem_title} - 求解出错\n\n{error_msg}",
                        current_output,
                        add_separator=True
                    )
            
            except Exception as e:
                error_msg = f"模型 {solver_model} 求解子问题出错：{str(e)}"
                logger.log_error(error_msg)
                yield from self._update_output(
                    f"# {subproblem_id}: {subproblem_title} - 求解出错\n\n{error_msg}",
                    current_output,
                    add_separator=True
                )
        
        # 添加总结
        yield from self._update_output(
            f"# 总结\n\n所有子问题已解答完成。",
            current_output,
            add_separator=True
        )

# 创建全局求解器实例
problem_solver = ProblemSolver()