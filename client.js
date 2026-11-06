window.__ModuleLoader__.load({
  id: 'lingming-memory',
  factory(require) {
    const React = require('react');
    const h = React.createElement;

    function runCommand(ctx, sessionId, line) {
      try {
        const remote = ctx && ctx.remote;
        if (remote && remote.commands && sessionId) {
          remote.commands.execute({ id: sessionId }, line, [], new AbortController().signal).catch(() => {});
        }
      } catch (e) {}
    }

    // 底部快捷开关
    function Toggle(props) {
      const [on, setOn] = React.useState(false);
      return h('button', {
        onClick: () => { setOn((v) => !v); runCommand(props.ctx, props.sessionId, '/lingming'); },
        role: 'switch', 'aria-checked': on,
        title: '灵铭记忆：开启后本地坐标寻址记忆边聊边学',
        style: {
          padding: '3px 10px', borderRadius: '999px',
          border: '1px solid var(--dsw-alias-border, #888)',
          background: on ? 'var(--dsw-alias-primary, #247bbf)' : 'transparent',
          color: on ? '#fff' : 'inherit',
          fontSize: '12px', cursor: 'pointer', lineHeight: '20px',
        },
      }, on ? '灵铭记忆·开' : '灵铭记忆·关');
    }

    // ============ 官方配置表单（configForms + plugins.bundle.config） ============
    const FIELDS = [
      { key: 'topK', label: '召回条数', hint: '每轮召回的历史记忆条数', kind: 'number', def: 3 },
      { key: 'embedServerUrl', label: '服务地址', hint: '本地记忆服务地址', kind: 'text', def: 'http://127.0.0.1:9000' },
      { key: 'python', label: 'Python 路径', hint: '带 torch 的 Python 可执行文件', kind: 'text', def: 'D:\\conda\\python.exe' },
      { key: 'enabled', label: '默认开启', hint: '可随时用 /lingming 切换', kind: 'bool', def: true },
      { key: 'd', label: '维度 d', hint: '模型维度（改了要重训坐标）', kind: 'number', def: 128 },
      { key: 'cMult', label: '横向倍数', hint: '坐标 bar 数 C = 横向倍数 × d', kind: 'number', def: 4 },
      { key: 'k', label: '稀疏 top-k', hint: '每条记忆点亮几个坐标 bar', kind: 'number', def: 8 },
      { key: 'capacity', label: '向量库容量', hint: '最多存多少条记忆（FIFO 淘汰）', kind: 'number', def: 10000 },
    ];

    const rowStyle = { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px', padding: '6px 0' };
    const labelStyle = { display: 'flex', flexDirection: 'column', gap: '2px', minWidth: 0 };
    const titleStyle = { fontSize: '13px', color: 'var(--dsw-alias-label-primary, inherit)' };
    const hintStyle = { fontSize: '12px', color: 'var(--dsw-alias-label-secondary, rgba(127,127,127,.8))' };
    const inputStyle = { width: '200px', padding: '4px 8px', borderRadius: '6px', border: '1px solid var(--dsw-alias-border, #888)', background: 'transparent', color: 'inherit', fontSize: '13px' };

    function Field(props) {
      const { f, value, writable, scope } = props;
      const current = value[f.key] !== undefined ? value[f.key] : f.def;
      const commit = (val) => { try { if (writable && scope && typeof scope.set === 'function') scope.set(f.key, val); } catch (e) {} };
      let control;
      if (f.kind === 'bool') {
        control = h('input', { type: 'checkbox', checked: current === true, disabled: !writable, onChange: (e) => commit(e.target.checked) });
      } else if (f.kind === 'number') {
        control = h('input', { type: 'number', value: current === undefined ? '' : current, style: inputStyle, disabled: !writable, onBlur: (e) => { const n = Number(e.target.value); if (!Number.isNaN(n)) commit(n); } });
      } else {
        control = h('input', { type: 'text', value: current === undefined ? '' : current, style: inputStyle, disabled: !writable, onBlur: (e) => commit(e.target.value) });
      }
      return h('div', { style: rowStyle },
        h('div', { style: labelStyle }, h('span', { style: titleStyle }, f.label), h('span', { style: hintStyle }, f.hint)),
        control,
      );
    }

    function LingmingConfig(props) {
      if (props.view !== 'page') return null;
      const scope = props.scope;
      if (!scope) return null;
      const snapshot = React.useSyncExternalStore(
        (listener) => scope.subscribe(listener),
        () => scope.getSnapshot(),
      );
      const value = snapshot.value || {};
      const writable = snapshot.writable === true;
      const ready = snapshot.status === 'ready';
      // 恢复默认：unset 所有字段 → 回到 base/默认值
      const resetDefaults = () => {
        try {
          if (!writable || typeof scope.unset !== 'function') return;
          for (const f of FIELDS) scope.unset(f.key);
        } catch (e) {}
      };
      return h('div', { style: { display: 'grid', gap: '2px', padding: '4px 0' } },
        ready ? null : h('div', { style: { fontSize: '12px', color: 'var(--dsw-alias-label-secondary, rgba(127,127,127,.8))', padding: '4px 0' } }, '配置不可用，容器还没就绪。'),
        ...FIELDS.map((f) => h(Field, { key: f.key, f, value, writable, scope })),
        h('div', { style: { display: 'flex', gap: '10px', marginTop: '10px' } },
          h('button', {
            onClick: resetDefaults,
            disabled: !writable,
            style: { padding: '5px 14px', borderRadius: '6px', border: '1px solid var(--dsw-alias-border, #888)', background: 'transparent', color: 'inherit', fontSize: '13px', cursor: 'pointer' },
          }, '恢复默认'),
        ),
      );
    }

    // 官方表单：configForms 服务取本插件表单，挂到 plugins.bundle.config 槽
    function mountConfigForm(ctx) {
      try {
        const forms = typeof ctx.get === 'function' ? ctx.get('configForms') : undefined;
        if (!forms || typeof forms.get !== 'function') return;
        let scope;
        for (const ns of ['lingming-memory', 'include:lingming-memory']) {
          try {
            const f = forms.get(ns);
            if (f && typeof f.getSnapshot === 'function') { scope = f; break; }
          } catch (e) {}
        }
        if (!scope) return;
        ctx.slots.inject('plugins.bundle.config', () => ctx.slots.register(
          { name: 'plugins.bundle.config', key: 'lingming-memory', inject: () => ({ scope }) },
          LingmingConfig,
        ));
      } catch (e) {}
    }

    return {
      inject: ['slots'],
      apply(ctx) {
        ctx.slots.inject('conversation.composer.dock', () => ctx.slots.register(
          { name: 'conversation.composer.dock', id: 'lingming-memory', order: 5 },
          (props) => h(Toggle, { ...props, ctx }),
        ));
        mountConfigForm(ctx);
      },
    };
  },
});
