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

    function Toggle({ ctx, sessionId }) {
      const [on, setOn] = React.useState(false);
      return h('button', {
        onClick: () => { setOn((v) => !v); runCommand(ctx, sessionId, '/lingming'); },
        role: 'switch', 'aria-checked': on,
        title: '灵铭记忆：开启后不走长上下文，改走坐标检索记忆',
        style: {
          padding: '3px 10px', borderRadius: '999px',
          border: '1px solid var(--dsw-alias-border, #888)',
          background: on ? 'var(--dsw-alias-primary, #247bbf)' : 'transparent',
          color: on ? '#fff' : 'inherit',
          fontSize: '12px', cursor: 'pointer', lineHeight: '20px',
        },
      }, on ? '灵铭记忆·开' : '灵铭记忆·关');
    }

    function Init({ ctx, sessionId }) {
      return h('button', {
        onClick: () => runCommand(ctx, sessionId, '/lingming-init'),
        title: '一键初始化坐标网络（用 API 打分生成可用坐标）',
        style: {
          padding: '3px 8px', borderRadius: '999px',
          border: '1px solid var(--dsw-alias-border, #888)',
          background: 'transparent', color: 'inherit',
          fontSize: '12px', cursor: 'pointer', lineHeight: '20px',
        },
      }, '初始化坐标');
    }

    return {
      inject: ['slots'],
      apply(ctx) {
        ctx.slots.inject('conversation.composer.dock', () => {
          const reg = (id, Cmp) => ctx.slots.register(
            { name: 'conversation.composer.dock', id, order: 5 },
            (props) => h(Cmp, { ...props, ctx }),
          );
          return () => {
            const d1 = reg('lingming-memory', Toggle);
            const d2 = reg('lingming-memory-init', Init);
            return () => { d1 && d1(); d2 && d2(); };
          };
        });
      },
    };
  },
});
