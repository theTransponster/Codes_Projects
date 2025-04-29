import java.awt.Color;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;

import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JColorChooser;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JRootPane;
import javax.swing.JScrollBar;
import javax.swing.JTextField;

public class FiguraControles extends JPanel implements ActionListener,MouseListener,MouseMotionListener {

	private FiguraPanel p;
	private JTextField tfAncho,
					   tfLargo,
					   tfAlto,
					   tfX,
					   tfY;
	private JButton btDibujar,
					btColor,
					btDibuMouse;
	private JRadioButton rbRectangulo,
						 rbCaja,
						 rbCuadrado;
	private JScrollBar sbFlechas;
	private Color color;
	private double ancho,
				   largo;
	private boolean mover;
	private int x,
				y,
				x2,
				y2;
	private Figura[] figuras={new Rectangulo(),new Caja(),new Cuadrado()};
	
	public FiguraControles(FiguraPanel p){
		super();
		this.setPreferredSize(new Dimension(150,600));
		this.p=p;
		this.setBackground(Color.white);
		this.color=Color.BLACK;
		this.ancho=0;
		this.largo=0;
		this.mover=false;
		this.add(new JLabel("Parametros para figuras"));
		this.add(new JLabel("Ancho"));
		this.tfAncho=new JTextField(8);
		this.add(this.tfAncho);
		
		this.add(new JLabel("Largo"));
		this.tfLargo=new JTextField(8);
		this.add(this.tfLargo);
		this.tfAlto=new JTextField(8);
		this.add(new JLabel("Ingresar Punto de Origen"));
		this.add(new JLabel("X: "));
		this.tfX=new JTextField(10);
		this.add(this.tfX);
		this.add(new JLabel("Y: "));
		this.tfY=new JTextField(10);
		this.add(this.tfY);
		
		this.rbCuadrado=new JRadioButton("Cuadrado    ");
		this.rbCaja=new JRadioButton("Caja");
		this.rbRectangulo=new JRadioButton("Rectangulo");
		
		ButtonGroup bg=new ButtonGroup();
		bg.add(this.rbCaja);
		bg.add(this.rbCuadrado);
		bg.add(this.rbRectangulo);
		
		this.add(this.rbRectangulo);
		this.add(this.rbCuadrado);
		this.add(this.rbCaja);
		
		this.rbRectangulo.setSelected(true);
		
		
		this.btDibujar=new JButton("Dibujar figura");
		this.btDibujar.addActionListener(this);
		this.add(this.btDibujar);
		this.btDibuMouse=new JButton("Dibujar con mouse");
		this.add(this.btDibuMouse);
		this.btDibuMouse.addActionListener(this);
		this.btColor=new JButton("Cambiar Color");
		this.add(this.btColor);
		this.btColor.addActionListener(this);
		this.p.addMouseListener(this);
		this.p.addMouseMotionListener(this);
		
	}


	@Override
	public void actionPerformed(ActionEvent e) {
		this.getRootPane().setDefaultButton(btDibujar);
		if(e.getSource()==this.btDibujar){
			this.mover=false;
			p.setCoordenadas(Integer.parseInt(this.tfX.getText()), Integer.parseInt(this.tfY.getText()));
			p.setModo(false);
			if(this.rbRectangulo.isSelected()){
				this.ancho=Double.parseDouble(this.tfAncho.getText());
				this.largo=Double.parseDouble(this.tfLargo.getText());
				this.figuras[0]=new Rectangulo(ancho,largo);
				p.setString(this.figuras[0].toString());
				p.setFigura(this.figuras[0]);
			}else if(this.rbCaja.isSelected()){
				String s=JOptionPane.showInputDialog("Introducir altura");
				double altura=Double.parseDouble(s);
				//JOptionPane.showMessageDialog(null, altura);
				this.ancho=Double.parseDouble(this.tfAncho.getText());
				this.largo=Double.parseDouble(this.tfLargo.getText());
				this.figuras[1]=new Caja(this.ancho,this.largo,altura);
				p.setString(this.figuras[1].toString());
				p.setFigura(this.figuras[1]);
				
			}else if(this.rbCuadrado.isSelected()){
				this.ancho=Double.parseDouble(this.tfAncho.getText());
				this.largo=Double.parseDouble(this.tfLargo.getText());
				this.figuras[2]=new Cuadrado(largo);
				p.setString(this.figuras[2].toString());
				p.setFigura(this.figuras[2]);
			}
			
		} else if(e.getSource()==this.btColor){
			this.color=JColorChooser.showDialog(this.p, "Seleccionar color", this.color);
			p.setColor(this.color);
		}
		else if(e.getSource()==this.btDibuMouse){
			this.mover=true;
			JOptionPane.showMessageDialog(null, "De click para coordenada incial \n Arrastre el mouse para dibujar" );
			p.setModo(true);
		}
	}


	@Override
	public void mouseClicked(MouseEvent e) {
		// TODO Auto-generated method stub
		if(this.mover==true){
			this.x=e.getX();
			this.y=e.getY();
			p.setCoordenadas(this.x, this.y);
		}
	}


	@Override
	public void mouseEntered(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}


	@Override
	public void mouseExited(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}


	@Override
	public void mousePressed(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}


	@Override
	public void mouseReleased(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}


	@Override
	public void mouseDragged(MouseEvent e) {
		// TODO Auto-generated method stub
		if(this.mover==true){
			this.x2=e.getX();
			this.y2=e.getY();
			p.setFinales(this.x2, this.y2);
		}
	}


	@Override
	public void mouseMoved(MouseEvent e) {
		// TODO Auto-generated method stub
		
	}
	
}
